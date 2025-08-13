using SafeTestsets: @safetestset

# check for filtered groups
# either via `--group=ALL` or through ENV["GROUP"]
const pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = uppercase(if isnothing(arg_id)
                            get(ENV, "GROUP", "ALL")
                        else
                            only(match(pat, ARGS[arg_id]).captures)
                        end)

"match files of the form `test_*.jl`, but exclude `*setup*.jl`"
istestfile(fn) = endswith(fn, ".jl") && startswith(basename(fn), "test_") &&
                 !contains(fn, "setup")

include("setup.jl")

@time begin
    # tests in groups based on folder structure
    for testgroup in filter(isdir, readdir(@__DIR__))
        if GROUP == "ALL" || GROUP == uppercase(testgroup)
            groupdir = joinpath(@__DIR__, testgroup)
            for file in filter(istestfile, readdir(groupdir))
                filename = joinpath(groupdir, file)
                @eval @safetestset $file begin
                    include($filename)
                end
            end
        end
    end

    # single files in top folder
    for file in filter(istestfile, readdir(@__DIR__))
        (file == basename(@__FILE__)) && continue # exclude this file to avoid infinite recursion
        @eval @safetestset $file begin
            include($file)
        end
    end
end
