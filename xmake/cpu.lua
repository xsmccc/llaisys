target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("../src/device/cpu/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end
    add_files("../src/ops/*/cpu/*.cpp")
    add_links("openblas")
    add_syslinks("pthread")
    on_install(function (target) end)
target_end()
