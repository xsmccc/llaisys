-- 天数智芯 TOPSRIDER GPU 构建配置
-- 天数智芯 BI-150 GPU 使用 TOPSRIDER SDK，通过 topscc 编译器编译 .cu 文件
-- TOPSRIDER SDK 提供 CUDA 兼容 API，.cu 文件使用标准 CUDA 语法

-- 自定义编译规则：使用 topscc 编译 .cu 文件
rule("topscc")
    set_extensions(".cu")
    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        local tops_home = os.getenv("TOPS_HOME") or "/opt/tops"
        local topscc = tops_home .. "/bin/topscc"
        local arch = get_config("tops-arch") or "gcu300"
        
        -- 收集头文件搜索路径
        local includedirs = {}
        for _, dir in ipairs(target:get("includedirs")) do
            table.insert(includedirs, "-I" .. dir)
        end
        -- 添加项目 include 目录
        table.insert(includedirs, "-I" .. os.projectdir() .. "/include")
        -- 添加 TOPSRIDER SDK include 目录
        table.insert(includedirs, "-I" .. tops_home .. "/include")
        
        -- 编译命令
        batchcmds:mkdir(path.directory(objectfile))
        batchcmds:vrunv(topscc, table.join(
            {"-c", "-std=c++17", "-fPIC", "-O2"},
            {"--cuda-gpu-arch=" .. arch},
            includedirs,
            {"-o", objectfile, sourcefile}
        ))
        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()

-- 天数智芯设备运行时静态库
target("llaisys-device-tianshu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("topscc")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("../src/device/tianshu/*.cu")
    
    on_install(function (target) end)
target_end()

-- 天数智芯算子静态库
target("llaisys-ops-tianshu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("topscc")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("../src/ops/*/tianshu/*.cu")
    
    on_install(function (target) end)
target_end()
