local function get_metax_sdk()
    local maca_path = os.getenv("MACA_PATH") or os.getenv("MACA_HOME")
    local mxcc = os.getenv("MXCC")

    if mxcc and not os.isfile(mxcc) then
        mxcc = nil
    end

    if not mxcc and maca_path then
        local candidate = path.join(maca_path, "bin", "mxcc")
        if os.isfile(candidate) then
            mxcc = candidate
        end
    end

    if not mxcc then
        for _, bindir in ipairs(path.splitenv(os.getenv("PATH") or "")) do
            local candidate = path.join(bindir, "mxcc")
            if os.isfile(candidate) then
                mxcc = candidate
                break
            end
        end
    end

    if not mxcc then
        maca_path = maca_path or "/opt/maca"
        mxcc = path.join(maca_path, "bin", "mxcc")
    end

    if not maca_path then
        maca_path = path.directory(path.directory(mxcc))
    end

    return maca_path, mxcc
end


rule("mxcc")
    set_extensions(".cu")
    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        local maca_path, mxcc = get_metax_sdk()
        local arch = get_config("maca-arch") or "mp_22"
        
        local includedirs = {}
        for _, dir in ipairs(target:get("includedirs")) do
            table.insert(includedirs, "-I" .. dir)
        end
        table.insert(includedirs, "-I" .. os.projectdir() .. "/include")
        table.insert(includedirs, "-I" .. maca_path .. "/include")
        
        batchcmds:mkdir(path.directory(objectfile))
        batchcmds:vrunv(mxcc, table.join(
            {"-c", "-std=c++17", "-fPIC", "-O2"},
            {"--offload-arch=" .. arch},
            includedirs,
            {"-o", objectfile, sourcefile}
        ))
        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()

target("llaisys-device-metax")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("mxcc")
    local maca_path = get_metax_sdk()
    add_includedirs(path.join(maca_path, "include"))
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("../src/device/metax/*.cu")
    
    on_install(function (target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("mxcc")
    local maca_path = get_metax_sdk()
    add_includedirs(path.join(maca_path, "include"))
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("../src/ops/*/metax/*.cu")
    
    on_install(function (target) end)
target_end()
