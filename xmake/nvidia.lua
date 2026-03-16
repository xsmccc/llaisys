-- NVIDIA CUDA Support Configuration for Xmake -- NVIDIA CUDA 支持配置

-- 为什么编译为静态库？
-- 1. 避免共享库加载时的依赖问题，简化部署
-- 2. 提高性能，减少运行时开销
-- 3. 确保与主程序的兼容性，避免符号冲突

target("llaisys-device-nvidia") -- 定义 NVIDIA 设备静态库目标
    set_kind("static") -- 生成静态库，而不是共享库
    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    
    -- Enable CUDA language
    add_languages("cuda") -- 启用 CUDA 语言支持
    -- 禁用 RDC（relocatable device code）减少额外显存占用
    set_values("cuda.rdc", false)
    set_policy("build.cuda.devlink", false)
    
    -- Set CUDA architecture - 使用可配置的 GPU 架构
    -- 通过 xmake f --cuda-arch=sm_XX 设置，常见值：
    --   sm_70 (V100, T4), sm_80 (A100, RTX30), sm_89 (RTX40), sm_90 (H100)
    local arch = get_config("cuda-arch") or "sm_89"
    add_cuflags("-arch=" .. arch, {tools = "nvcc"}) -- 设置 CUDA 架构
    add_cuflags("-Xcompiler -fPIC", {tools = "nvcc"}) -- 透传 C++ 编译器参数
    
    -- Add PIC flag for Linux （position independent code）位置独立代码
    -- 共享库需要PIC 以支持地址随机化
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 生成与位置无关代码
    end

    -- Add CUDA source files
    add_files("../src/device/nvidia/*.cu") -- 添加 NVIDIA 设备 CUDA 源码
    
    -- Link with CUDA Runtime library
    -- 连接CUDA Runtime库
    add_syslinks("cudart", "cublas", "cublasLt", {tools = "nvcc"}) -- 链接 CUDA 运行时库 + cuBLAS

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义

target("llaisys-ops-nvidia") -- 定义 NVIDIA 算子静态库目标
    set_kind("static") -- 生成静态库
    add_deps("llaisys-tensor") -- 依赖 Tensor 层
    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    
    -- Enable CUDA language
    add_languages("cuda") -- 启用 CUDA 语言支持
    -- 禁用 RDC（relocatable device code）减少额外显存占用
    set_values("cuda.rdc", false)
    set_policy("build.cuda.devlink", false)
    
    -- Set CUDA architecture - 使用可配置的 GPU 架构
    local arch = get_config("cuda-arch") or "sm_89"
    add_cuflags("-arch=" .. arch, {tools = "nvcc"}) -- 设置 CUDA 架构
    add_cuflags("-Xcompiler -fPIC", {tools = "nvcc"}) -- 透传 C++ 编译器参数
    
    -- Add PIC flag for Linux
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
    end

    -- Add CUDA operator implementation files
    add_files("../src/ops/*/nvidia/*.cu") -- 添加 NVIDIA 算子 CUDA 源码

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义
