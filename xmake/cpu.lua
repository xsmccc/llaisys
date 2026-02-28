target("llaisys-device-cpu") -- 定义 CPU 设备静态库目标
    set_kind("static") -- 生成静态库
    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
    end

    add_files("../src/device/cpu/*.cpp") -- 添加 CPU 设备源码

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义

target("llaisys-ops-cpu") -- 定义 CPU 算子静态库目标
    set_kind("static") -- 生成静态库
    add_deps("llaisys-tensor") -- 依赖 Tensor 层
    set_languages("cxx17") -- 使用 C++17 标准
    set_warnings("all", "error") -- 警告全开并视为错误
    if not is_plat("windows") then -- 非 Windows 平台
        add_cxflags("-fPIC", "-Wno-unknown-pragmas") -- 位置无关代码与忽略未知 pragma
        add_cxflags("-fopenmp") -- 启用 OpenMP 支持
        add_ldflags("-fopenmp") -- 链接 OpenMP 库
    end

    add_files("../src/ops/*/cpu/*.cpp") -- 添加 CPU 算子源码
    
    -- 添加 OpenBLAS 支持
    add_links("openblas") -- 链接 OpenBLAS 库
    add_syslinks("pthread") -- OpenBLAS 需要 pthread

    on_install(function (target) end) -- 安装阶段占位
target_end() -- 结束目标定义

