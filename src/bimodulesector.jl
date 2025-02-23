struct BimoduleSector{Name} <: Sector
    i::Int
    j::Int
    label::Int
end

const A4Object = BimoduleSector{:A4}

# Utility implementations
# -----------------------
function Base.isless(a::I, b::I) where {I<:BimoduleSector}
    return isless((a.i, a.j, a.label), (b.i, b.j, b.label))
end
Base.hash(a::BimoduleSector, h::UInt) = hash(a.i, hash(a.j, hash(a.label, h)))
Base.convert(::Type{BimoduleSector{Name}}, d::NTuple{3,Int}) = BimoduleSector{Name}(d...)

Base.IteratorSize(::Type{SectorValues{<:BimoduleSector}}) = Base.SizeUnknown()

# TODO: generalize?
# TODO: numlabels?
function Base.iterate(iter::SectorValues{A4Object}, (I, label)=(1, 1))
    I > 12 * 12 && return nothing
    i, j = CartesianIndices((12, 12))[I].I
    maxlabel = numlabels(A4Object, i, j)
    return if label > maxlabel
        iterate(iter, (I + 1, 1))
    else
        A4Object(i, j, label), (I, label + 1)
    end
end

TensorKitSectors.FusionStyle(::Type{A4Object}) = GenericFusion()
TensorKitSectors.BraidingStyle(::Type{A4Object}) = NoBraiding()

function TensorKitSectors.:⊗(a::A4Object, b::A4Object)
    @assert a.j == b.i
    return Iterators.filter(c -> NSymbol(a, b, c) > 0,
                            map(label -> A4Object(a.i, b.j, label),
                                numlabels(A4Object, a.i, b.j)))
end

# TODO: can I assume A4Irrep(i, i, 1) is identity?
function Base.conj(a::A4Object)
    i, j = a.i, a.j
    label = findfirst(x -> Nsymbol(a, A4Irrep(j, i, x), A4Irrep(i, i, 1)) == 1,
                      1:numlabels(A4Object, j, i))
    return A4Object(j, i, label)
end

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

const A4_Ncache = extract_Nsymbol(A4Object)
function TensorKitSectors.Nsymbol(a::I, b::I, c::I) where {I<:A4Object}
    # TODO: should this error or return 0?
    (a.j == b.i && a.i == c.i && b.j == c.j) ||
        throw(ArgumentError("invalid fusion channel"))
    i, j, k = a.i, a.j, b.j
    return get(A4_Ncache[i, j, k], (a.label, b.label, c.label), 0)
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

const A4_Fcache = extract_Fsymbol(A4Object)
function TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I,
                                  f::I) where {I<:A4Object}
    # TODO: should this error or return 0?
    (a.j == b.i && b.j == c.i && a.i == d.i && c.j == d.j &&
     a.i == e.i && b.j == e.j && f.i == a.j && f.j == c.j) ||
        throw(ArgumentError("invalid fusion channel"))

    i, j, k, l = a.i, a.j, b.j, c.j
    return get(A4_Fcache[i, j, k, l],
               (a.label, b.label, c.label, d.label, e.label, f.label)) do
        return zeros(sectorscalartype(A4Object),
                     (Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f),
                      Nsymbol(a, f, d)))
    end
end
