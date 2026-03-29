target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_languages("cuda")
    set_values("cuda.rdc", false)
    set_policy("build.cuda.devlink", false)
    
    local arch = get_config("cuda-arch") or "sm_89"
    add_cuflags("-arch=" .. arch, {tools = "nvcc"})
    add_cuflags("-Xcompiler -fPIC", {tools = "nvcc"})
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cu")
    add_syslinks("cudart", "cublas", "cublasLt", {tools = "nvcc"})

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_languages("cuda")
    set_values("cuda.rdc", false)
    set_policy("build.cuda.devlink", false)
    
    local arch = get_config("cuda-arch") or "sm_89"
    add_cuflags("-arch=" .. arch, {tools = "nvcc"})
    add_cuflags("-Xcompiler -fPIC", {tools = "nvcc"})
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
