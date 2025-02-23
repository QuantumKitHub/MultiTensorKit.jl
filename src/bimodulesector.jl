struct BimoduleSector{Name} <: Sector
    i::Int
    j::Int
    label::Int
end

const A4Object = BimoduleSector{:A4}

# Data from files
# ---------------
const artifact_path = joinpath(artifact"fusiondata", "MultiTensorKit.jl-data-v0.1.0")

function extract_Nsymbol(::Type{A4Object})
    filename = joinpath(artifact_path, "A4", "Nsymbol.json")
    isfile(filename) || throw(LoadError(filename, 0, "Nsymbol file not found for $Name"))
    json_string = read(filename, String)
    Narray = copy(JSON3.read(json_string))
    return map(reshape(Narray, 12, 12, 12)) do x
        y = Dict{NTuple{3,Int},Int}()
        for (k, v) in x
            a, b, c = parse.(Int, split(string(k)[2:(end - 1)], ", "))
            y[(a, b, c)] = v
        end
    end
end

let Ncache = extract_Nsymbol(A4Object)
    function TensorKitSectors.Nsymbol(a::I, b::I, c::I) where {I<:A4Object}
        # TODO: should this error or return 0?
        (a.j == b.i && a.i == c.i && b.j == c.j) ||
            throw(ArgumentError("invalid fusion channel"))
        i, j, k = a.i, a.j, b.j
        return get(Ncache[i, j, k], (a.label, b.label, c.label), 0)
    end
end

function extract_Fsymbol(::Type{A4Object})
    return mapreduce(vcat, 1:12) do i
        filename = joinpath(artifact_path, "A4", "Fsymbol$i.json")
        @assert isfile(filename)
        json_string = read(filename, String)
        Farray_part = copy(JSON3.read(json_string))
        return map(reshape(Farray_part, 12, 12, 12, 12)) do x
            y = Dict{NTuple{6,Int},ComplexF64}()
            for (k, v) in x
                a, b, c, d, e, f = parse.(Int, split(string(k)[2:(end - 1)], ", "))
                y[(a, b, c, d, e, f)] = v
            end
        end
    end
end

let Fcache = extract_Fsymbol(A4Object)
    function TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I,
                                      f::I) where {I<:A4Object}
        # TODO: should this error or return 0?
        (a.j == b.i && b.j == c.i && a.i == d.i && c.j == d.j &&
         a.i == e.i && b.j == e.j && f.i == a.j && f.j == c.j) ||
            throw(ArgumentError("invalid fusion channel"))

        i, j, k, l = a.i, a.j, b.j, c.j
        return get(Fcache[i, j, k, l],
                   (a.label, b.label, c.label, d.label, e.label, f.label)) do
            return zeros(sectorscalartype(A4Object),
                         (Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f),
                          Nsymbol(a, f, d)))
        end
    end
end
