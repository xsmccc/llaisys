#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

namespace llaisys::models::qwen2 {

/**
 * CUDA Graph Manager for Decode Optimization
 *
 * Two modes:
 * 1. Re-capture: Re-capture the graph every token, use cudaGraphExecUpdate
 *    to avoid re-instantiation when topology is unchanged.
 * 2. Static capture (default): Capture once, replay forever. Kernels read
 *    changing parameters (start_pos, total_len) from device memory pointers.
 *    Only H2D updates needed before each launch (~17μs vs ~800μs re-capture).
 */
class CudaGraphManager {
public:
    CudaGraphManager() = default;
    ~CudaGraphManager() { cleanup(); }

    CudaGraphManager(const CudaGraphManager&) = delete;
    CudaGraphManager& operator=(const CudaGraphManager&) = delete;

    // ========== Re-capture mode (legacy) ==========

    template <typename F>
    void captureAndLaunch(cudaStream_t stream, F&& forward) {
        if (warmup_remaining_ > 0) {
            forward();
            warmup_remaining_--;
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        forward();
        cudaGraph_t graph = nullptr;
        cudaError_t cap_err = cudaStreamEndCapture(stream, &graph);

        if (cap_err == cudaSuccess && graph) {
            size_t numNodes = 0;
            cudaGraphGetNodes(graph, nullptr, &numNodes);
            if (launch_count_ == 0)
                std::cerr << "[CudaGraph] Graph has " << numNodes << " nodes" << std::endl;
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        if (cap_err != cudaSuccess || graph == nullptr) {
            std::cerr << "[CudaGraph] Capture failed: " << cudaGetErrorString(cap_err)
                      << " — falling back to direct execution" << std::endl;
            forward();
            return;
        }

        if (exec_) {
            cudaGraphExecUpdateResultInfo info = {};
            cudaError_t upd_err = cudaGraphExecUpdate(exec_, graph, &info);
            if (upd_err != cudaSuccess || info.result != cudaGraphExecUpdateSuccess) {
                cudaGraphExecDestroy(exec_);
                exec_ = nullptr;
                cudaError_t inst_err = cudaGraphInstantiate(&exec_, graph, 0);
                if (inst_err != cudaSuccess) {
                    std::cerr << "[CudaGraph] Re-instantiate failed: "
                              << cudaGetErrorString(inst_err) << std::endl;
                    cudaGraphDestroy(graph);
                    forward();
                    return;
                }
                reinstantiate_count_++;
            }
        } else {
            cudaError_t inst_err = cudaGraphInstantiate(&exec_, graph, 0);
            if (inst_err != cudaSuccess) {
                std::cerr << "[CudaGraph] Instantiate failed: "
                          << cudaGetErrorString(inst_err) << std::endl;
                cudaGraphDestroy(graph);
                forward();
                return;
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        cudaGraphDestroy(graph);
        cudaGraphLaunch(exec_, stream);
        launch_count_++;

        auto t3 = std::chrono::high_resolution_clock::now();

        total_capture_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_update_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        total_launch_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    }

    // ========== Static capture mode ==========

    /**
     * Static capture: capture graph once, replay on every subsequent call.
     *
     * @param stream     CUDA stream
     * @param setup_fn   Pre-launch setup (H2D copies to update device params).
     *                   Runs on the stream BEFORE graph launch. NOT captured.
     * @param forward_fn The forward pass lambda. Only called once (for capture).
     *                   Must use device pointers for changing parameters.
     */
    template <typename SetupF, typename ForwardF>
    void staticLaunch(cudaStream_t stream, SetupF&& setup_fn, ForwardF&& forward_fn) {
        if (warmup_remaining_ > 0) {
            // Warmup: run directly (lets cuBLAS/kernel caches initialize)
            setup_fn();
            forward_fn();
            warmup_remaining_--;
            return;
        }

        if (!static_captured_) {
            // First real call: capture the graph
            auto t0 = std::chrono::high_resolution_clock::now();

            // Run setup on stream (H2D params)
            setup_fn();

            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal); // 开始录制
            forward_fn();   //不执行，只记录参数
            cudaGraph_t graph = nullptr;
            cudaError_t cap_err = cudaStreamEndCapture(stream, &graph); // 结束录制

            if (cap_err != cudaSuccess || graph == nullptr) {
                std::cerr << "[CudaGraph/Static] Capture failed: "
                          << cudaGetErrorString(cap_err)
                          << " — falling back to direct execution" << std::endl;
                // Fallback: just run directly
                forward_fn();
                return;
            }

            size_t numNodes = 0;
            cudaGraphGetNodes(graph, nullptr, &numNodes);
            std::cerr << "[CudaGraph/Static] Graph captured: " << numNodes
                      << " nodes" << std::endl;

            cudaError_t inst_err = cudaGraphInstantiate(&exec_, graph, 0); // 实例化
            cudaGraphDestroy(graph);    // 销毁
            if (inst_err != cudaSuccess) {
                std::cerr << "[CudaGraph/Static] Instantiate failed: "
                          << cudaGetErrorString(inst_err) << std::endl;
                exec_ = nullptr;
                forward_fn();
                return;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            total_capture_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            static_captured_ = true;    // 已录制
            ever_static_ = true;

            // Launch the captured graph (first real output)
            cudaGraphLaunch(exec_, stream); // 第一次launch
            launch_count_++;
            return;
        }

        // Subsequent calls: update params via H2D, then launch
        auto t0 = std::chrono::high_resolution_clock::now();
        setup_fn(); // 更新参数
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(exec_, stream);
        auto t2 = std::chrono::high_resolution_clock::now();
        launch_count_++;

        total_update_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_launch_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }

    bool isStaticCaptured() const { return static_captured_; }

    void reset() {
        cleanup();
        warmup_remaining_ = WARMUP_TOKENS;
        static_captured_ = false;
    }

    void cleanup() {
        if (exec_) {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
    }

    void printStats() const {
        if (launch_count_ == 0) return;
        const char* stats_env = std::getenv("LLAISYS_CUDA_GRAPH_STATS");
        if (!stats_env || std::string(stats_env) != "1") return;

        if (ever_static_) {
            // For static mode: capture is one-time cost, report setup+launch avg
            size_t replay_count = launch_count_ > 1 ? launch_count_ - 1 : 1;
            double avg_setup = (double)total_update_us_ / replay_count;
            double avg_launch = (double)total_launch_us_ / replay_count;
            std::cerr << "[CudaGraph/Static] launches=" << launch_count_
                      << " capture_once=" << total_capture_us_ << "us"
                      << " avg_setup=" << avg_setup << "us"
                      << " avg_launch=" << avg_launch << "us"
                      << " avg_total=" << (avg_setup + avg_launch) << "us"
                      << std::endl;
        } else {
            double avg_cap = (double)total_capture_us_ / launch_count_;
            double avg_upd = (double)total_update_us_ / launch_count_;
            double avg_lch = (double)total_launch_us_ / launch_count_;
            std::cerr << "[CudaGraph] launches=" << launch_count_
                      << " re-inst=" << reinstantiate_count_
                      << " avg_capture=" << avg_cap << "us"
                      << " avg_update=" << avg_upd << "us"
                      << " avg_launch=" << avg_lch << "us"
                      << " total_overhead=" << (avg_cap + avg_upd + avg_lch) << "us"
                      << std::endl;
        }
    }

private:
    static constexpr int WARMUP_TOKENS = 1;

    cudaGraphExec_t exec_ = nullptr;
    int warmup_remaining_ = WARMUP_TOKENS;
    int launch_count_ = 0;
    int reinstantiate_count_ = 0;
    bool static_captured_ = false;
    bool ever_static_ = false;
    long long total_capture_us_ = 0;
    long long total_update_us_ = 0;
    long long total_launch_us_ = 0;
};

}  // namespace llaisys::models::qwen2
