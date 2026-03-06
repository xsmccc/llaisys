-- MetaX MACA GPU 构建配置
-- 沐曦 C500 GPU 使用 MACA SDK，通过 mxcc 编译器编译 .cu 文件
-- MACA SDK 提供 CUDA 兼容 API，.cu 文件使用标准 CUDA 语法

-- 自定义编译规则：使用 mxcc 编译 .cu 文件
rule("mxcc")
    set_extensions(".cu")
    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        local maca_path = os.getenv("MACA_PATH") or "/opt/maca"
        local mxcc = maca_path .. "/bin/mxcc"
        local arch = get_config("maca-arch") or "mp_22"
        
        -- 收集头文件搜索路径
        local includedirs = {}
        for _, dir in ipairs(target:get("includedirs")) do
            table.insert(includedirs, "-I" .. dir)
        end
        -- 添加项目 include 目录
        table.insert(includedirs, "-I" .. os.projectdir() .. "/include")
        -- 添加 MACA SDK include 目录
        table.insert(includedirs, "-I" .. maca_path .. "/include")
        
        -- 编译命令
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

-- MetaX 设备运行时静态库
target("llaisys-device-metax")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("mxcc")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("../src/device/metax/*.cu")
    
    on_install(function (target) end)
target_end()

-- MetaX 算子静态库
target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("mxcc")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("../src/ops/*/metax/*.cu")
    
    on_install(function (target) end)
target_end()
