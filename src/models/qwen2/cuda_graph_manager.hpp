#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

namespace llaisys::models::qwen2 {

/**
 * CUDA Graph Manager for Decode Optimization
 *
 * Strategy: Re-capture + cudaGraphExecUpdate every decode token.
 *
 * Why re-capture every token?
 *   current_pos_ (KV cache offset) changes each step → kernel args differ.
 *   During capture, kernels are NOT executed (just recorded) — cost ≈ 1μs/kernel.
 *   500 kernel captures ≈ 0.5ms, vs 500 real launches ≈ 3ms → 2.5ms saved/token.
 *
 * cudaGraphExecUpdate:
 *   If graph topology matches (same kernels, same grid/block dims), it updates
 *   kernel arguments in the existing exec without re-instantiation (~20μs).
 *   Re-instantiation only happens when grid dims change (e.g., FlashDecoding
 *   every ~256 tokens).
 */
class CudaGraphManager {
public:
    CudaGraphManager() = default;
    ~CudaGraphManager() { cleanup(); }

    CudaGraphManager(const CudaGraphManager&) = delete;
    CudaGraphManager& operator=(const CudaGraphManager&) = delete;

    /**
     * Capture a forward pass, update/instantiate the graph, and launch it.
     *
     * @param stream    The CUDA stream to capture on and launch from.
     * @param forward   Lambda/callable that runs the complete forward pass
     *                  (embedding → layers → norm → lm_head → argmax).
     *                  During capture, these kernels are recorded (not executed).
     *                  After capture, the graph is launched to execute them.
     *
     * On first call: instantiates the graph exec.
     * On subsequent calls: uses cudaGraphExecUpdate (fast arg-only update).
     * If topology changes: falls back to re-instantiation.
     */
    template <typename F>
    void captureAndLaunch(cudaStream_t stream, F&& forward) {
        if (warmup_remaining_ > 0) {
            forward();
            warmup_remaining_--;
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // --- Capture ---
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        forward();
        cudaGraph_t graph = nullptr;
        cudaError_t cap_err = cudaStreamEndCapture(stream, &graph);

        auto t1 = std::chrono::high_resolution_clock::now();

        if (cap_err != cudaSuccess || graph == nullptr) {
            std::cerr << "[CudaGraph] Capture failed: " << cudaGetErrorString(cap_err)
                      << " — falling back to direct execution" << std::endl;
            forward();
            return;
        }

        // --- Update or Instantiate ---
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

        // --- Launch ---
        cudaGraphLaunch(exec_, stream);
        launch_count_++;

        auto t3 = std::chrono::high_resolution_clock::now();

        // Accumulate timing stats (microseconds)
        total_capture_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_update_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        total_launch_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    }

    void reset() {
        cleanup();
        warmup_remaining_ = WARMUP_TOKENS;
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

private:
    static constexpr int WARMUP_TOKENS = 1;

    cudaGraphExec_t exec_ = nullptr;
    int warmup_remaining_ = WARMUP_TOKENS;
    int launch_count_ = 0;
    int reinstantiate_count_ = 0;
    long long total_capture_us_ = 0;
    long long total_update_us_ = 0;
    long long total_launch_us_ = 0;
};

}  // namespace llaisys::models::qwen2
