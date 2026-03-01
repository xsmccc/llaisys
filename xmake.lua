add_rules("mode.debug", "mode.release") -- 添加调试/发布构建模式
set_encodings("utf-8") -- 设置源码/输出编码为 UTF-8

add_includedirs("include") -- 添加头文件全局搜索路径

-- CPU --
includes("xmake/cpu.lua") -- 引入 CPU 相关构建配置

-- NVIDIA --
option("nv-gpu") -- 定义是否启用 NVIDIA GPU 的构建开关
    set_default(false) -- 默认关闭 GPU 支持
    set_showmenu(true) -- 允许在命令行菜单中显示该选项
    set_description("Whether to compile implementations for Nvidia GPU") -- 选项说明
option_end() -- 结束选项定义

option("cuda-arch") -- CUDA GPU 架构版本
    set_default("sm_89") -- 默认 sm_89 (RTX 40 系列)，A100 用 sm_80，V100 用 sm_70
    set_showmenu(true)
    set_description("CUDA GPU architecture (e.g. sm_70, sm_80, sm_89)")
option_end()

if has_config("nv-gpu") then -- 如果启用 nv-gpu 选项
    add_defines("ENABLE_NVIDIA_API") -- 添加宏定义以开启 NVIDIA API
    includes("xmake/nvidia.lua") -- 引入 NVIDIA 相关构建配置
end -- 结束条件判断

target("llaisys-utils") -- 定义工具库目标
    set_kind("static") -- 生成静态库

    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
    end

    add_files("src/utils/*.cpp") -- 添加 utils 源文件

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义


target("llaisys-device") -- 定义设备层静态库
    set_kind("static") -- 生成静态库
    add_deps("llaisys-utils") -- 依赖工具库
    add_deps("llaisys-device-cpu") -- 依赖 CPU 设备实现
    
    -- Note: llaisys-device-nvidia is not added here to avoid duplicate compilation
    -- CUDA files will be directly compiled into the shared library

    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
    end

    add_files("src/device/*.cpp") -- 添加设备通用源码

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义

target("llaisys-core") -- 定义核心层静态库
    set_kind("static") -- 生成静态库
    add_deps("llaisys-utils") -- 依赖工具库
    add_deps("llaisys-device") -- 依赖设备层

    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
    end

    add_files("src/core/*/*.cpp") -- 添加 core 源码

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义

target("llaisys-tensor") -- 定义 Tensor 静态库
    set_kind("static") -- 生成静态库
    add_deps("llaisys-core") -- 依赖核心层

    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
    end

    add_files("src/tensor/*.cpp") -- 添加 Tensor 源码

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义

target("llaisys-ops") -- 定义算子层静态库
    set_kind("static") -- 生成静态库
    add_deps("llaisys-ops-cpu") -- 依赖 CPU 算子实现
    
    if has_config("nv-gpu") then -- 若启用 GPU
        add_deps("llaisys-ops-nvidia") -- 依赖 NVIDIA 算子实现
    end

    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
    end
    
    add_files("src/ops/*/*.cpp") -- 添加算子源码

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义

target("llaisys") -- 定义最终共享库目标
    set_kind("shared") -- 生成共享库
    add_deps("llaisys-utils") -- 依赖工具库
    add_deps("llaisys-core") -- 依赖核心层
    add_deps("llaisys-tensor") -- 依赖 Tensor 层
    add_deps("llaisys-ops") -- 依赖算子层

    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    add_files("src/llaisys/*.cc") -- 添加对外接口实现
    add_files("src/models/qwen2/*.cpp") -- 添加 Qwen2 模型实现
    add_files("src/device/*.cpp") -- 添加设备抽象层通用实现
    set_installdir(".") -- 安装目录为当前目录
    
    -- 添加 OpenMP 和 OpenBLAS 链接（共享库需要）
    if not is_plat("windows") then
        add_ldflags("-fopenmp") -- 链接 OpenMP
        add_syslinks("gomp") -- 显式链接 GNU OpenMP 库
    end
    add_links("openblas", "pthread") -- 链接 OpenBLAS 和 pthread

    -- Add CUDA support if enabled
    -- 检查是否启用了CUDA支持
    if has_config("nv-gpu") then -- 若启用 GPU
        add_languages("cuda") -- 启用 CUDA 语言
        -- 禁用 RDC（relocatable device code）减少额外显存占用
        set_values("cuda.rdc", false)
        set_policy("build.cuda.devlink", false)
        add_files("src/device/nvidia/*.cu") -- 添加 NVIDIA CUDA 源码
        -- 使用可配置的 GPU 架构（通过 xmake f --cuda-arch=sm_XX 设置）
        local arch = get_config("cuda-arch") or "sm_89"
        add_cuflags("-arch=" .. arch, "-Xcompiler -fPIC", {tools = "nvcc"}) -- 设置架构与 PIC
        -- 自动检测 CUDA 安装路径
        local cuda_path = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME") or "/usr/local/cuda"
        add_linkdirs(cuda_path .. "/lib64")  -- CUDA 库路径
        add_links("cudart", "cublas") -- 连接 libcudart.so (CUDA Runtime) + cuBLAS
        add_rpathdirs(cuda_path .. "/lib64") -- 运行时库查找路径
    else
        -- Only CPU device when CUDA is not enabled
        add_deps("llaisys-device") -- 仅依赖 CPU 设备实现
    end
    
    -- Always add CPU device files
    add_files("src/device/cpu/*.cpp") -- 始终编译 CPU 设备源码

    after_install(function (target) -- 安装后执行
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..") -- 打印提示
        if is_plat("windows") then -- Windows 平台
            os.cp("bin/*.dll", "python/llaisys/libllaisys/") -- 复制 DLL
        end
        if is_plat("linux") then -- Linux 平台
            os.cp("lib/*.so", "python/llaisys/libllaisys/") -- 复制 SO
        end
    end) -- 结束安装后操作
target_end() -- 结束目标定义