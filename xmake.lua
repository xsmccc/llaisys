add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

local function get_metax_sdk_paths()
    local maca_path = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or "/opt/maca"
    if not os.isdir(maca_path) and os.getenv("MXCC") and os.isfile(os.getenv("MXCC")) then
        maca_path = path.directory(path.directory(os.getenv("MXCC")))
    end

    local libdir = path.join(maca_path, "lib64")
    if not os.isdir(libdir) then
        libdir = path.join(maca_path, "lib")
    end

    return maca_path, libdir
end

add_includedirs("include")

-- CPU
includes("xmake/cpu.lua")

-- NVIDIA
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("cuda-arch")
    set_default("sm_89")
    set_showmenu(true)
    set_description("CUDA GPU architecture (e.g. sm_70, sm_80, sm_89)")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

-- MetaX MACA
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU (MACA)")
option_end()

option("maca-arch")
    set_default("mp_22")
    set_showmenu(true)
    set_description("MACA GPU architecture (e.g. mp_21, mp_22)")
option_end()

if has_config("metax-gpu") then
    add_defines("ENABLE_METAX_API")
    includes("xmake/metax.lua")
end


target("llaisys-utils")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/utils/*.cpp")
    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    
    -- CUDA files will be directly compiled into the shared library
    -- to avoid duplicate compilation

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/device/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/core/*/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/tensor/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    if has_config("metax-gpu") then
        add_deps("llaisys-ops-metax")
    end


    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/ops/*/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    add_files("src/models/qwen2/*.cpp")
    add_files("src/models/llama3/*.cpp")
    add_files("src/device/*.cpp")
    set_installdir(".")
    
    if not is_plat("windows") then
        add_ldflags("-fopenmp")
        add_syslinks("gomp")
    end
    add_links("openblas", "pthread")

    -- NVIDIA CUDA
    if has_config("nv-gpu") then
        add_languages("cuda")
        set_values("cuda.rdc", false)
        set_policy("build.cuda.devlink", false)
        add_files("src/device/nvidia/*.cu")
        local arch = get_config("cuda-arch") or "sm_89"
        add_cuflags("-arch=" .. arch, "-Xcompiler -fPIC", {tools = "nvcc"})
        local cuda_path = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME") or "/usr/local/cuda"
        add_linkdirs(cuda_path .. "/lib64")
        add_links("cudart", "cublas", "cublasLt")
        add_rpathdirs(cuda_path .. "/lib64")
    end

    -- MetaX MACA
    if has_config("metax-gpu") then
        add_rules("mxcc")
        add_files("src/device/metax/*.cu")
        add_cxflags("-fPIC")
        local maca_path, maca_libdir = get_metax_sdk_paths()
        add_includedirs(path.join(maca_path, "include"))
        add_linkdirs(maca_libdir)
        add_links("macart", "macablas")
        add_rpathdirs(maca_libdir)
    end


    if not has_config("nv-gpu") and not has_config("metax-gpu") then
        add_deps("llaisys-device")
    end
    
    add_files("src/device/cpu/*.cpp")

    after_install(function (target)
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
