#include "context.hpp"
#include "../../utils.hpp"
#include <thread>

namespace llaisys::core {

Context::Context() {
    // 列举设备类型（优先非 CPU，最后 CPU 作后备）
    std::vector<llaisysDeviceType_t> device_typs;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_typs.push_back(static_cast<llaisysDeviceType_t>(i));// NVIDIA 等
    }
    device_typs.push_back(LLAISYS_DEVICE_CPU);// CPU 最后

    // 遍历每个设备类型，查询有多少个该类型的设备
    // 激活第一个支持的设备，如果没有，则激活CPU
    for (auto device_type : device_typs) {
        const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);
        int device_count = api_->get_device_count();    //调用API获取设备数量
        std::vector<Runtime *> runtimes_(device_count);
        // 为该设备类型的每一个设备创建Runtime对象
        for (int device_id = 0; device_id < device_count; device_id++) {

            if (_current_runtime == nullptr) {
                // 创建第一个可用的 Runtime 并激活（作为默认）
                auto runtime = new Runtime(device_type, device_id);
                runtime->_activate();
                runtimes_[device_id] = runtime;
                _current_runtime = runtime;
            }
        }
        _runtime_map[device_type] = runtimes_;
    }
}

Context::~Context() {
    // Destroy current runtime first.
    delete _current_runtime;

    for (auto &runtime_entry : _runtime_map) {
        std::vector<Runtime *> runtimes = runtime_entry.second;
        for (auto runtime : runtimes) {
            if (runtime != nullptr && runtime != _current_runtime) {
                runtime->_activate();
                delete runtime;
            }
        }
        runtimes.clear();
    }
    _current_runtime = nullptr;
    _runtime_map.clear();
}

// 设备切换的核心逻辑
void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    // 检查是否已经在设备上
    if (_current_runtime == nullptr ||
         _current_runtime->deviceType() != device_type ||
          _current_runtime->deviceId() != device_id) {
        
        // 从map中查找该设备的Runtime
        auto runtimes = _runtime_map[device_type];
        CHECK_ARGUMENT((size_t)device_id < runtimes.size() && device_id >= 0, "invalid device id");
        // 如果之前有激活的 Runtime，失活它
        if (_current_runtime != nullptr) {
            _current_runtime->_deactivate();
        }

        // 如果目标设备的 Runtime 不存在，创建它
        if (runtimes[device_id] == nullptr) {
            runtimes[device_id] = new Runtime(device_type, device_id);
        }

        // 激活目标 Runtime
        runtimes[device_id]->_activate();
        _current_runtime = runtimes[device_id];
    }
}

// 获取当前Runtime
Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

// 全局函数
Context &context() {
    thread_local Context thread_context;    //每个线程独立的context
    return thread_context;
}

} // namespace llaisys::core
