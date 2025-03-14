struct BimoduleSector{Name} <: Sector
    i::Int
    j::Int
    label::Int
end
BimoduleSector{Name}(data::NTuple{3,Int}) where {Name} = BimoduleSector{Name}(data...)
#TODO: place bound on what BimoduleSectors can be made
# currently you can define A4Object(anyint, anyint, anyint)
const A4Object = BimoduleSector{:A4}
# function A4Object(i::Int, j::Int, label::Int)
#     i <= 12 && j <= 12 || throw(DomainError("object outside the matrix"))
#     return BimoduleSector{A4Object}(i, j, label)
# end

# Utility implementations
# -----------------------
function Base.isless(a::I, b::I) where {I<:BimoduleSector}
    return isless((a.i, a.j, a.label), (b.i, b.j, b.label))
end
Base.hash(a::BimoduleSector, h::UInt) = hash(a.i, hash(a.j, hash(a.label, h)))
function Base.convert(::Type{BimoduleSector{Name}}, d::NTuple{3,Int}) where {Name}
    return BimoduleSector{Name}(d...)
end

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
    Ncache = _get_Ncache(A4Object)[a.i, a.j, b.j]
    return A4Object[A4Object(a.i, b.j, c_l)
                    for (a_l, b_l, c_l) in keys(Ncache)
                    if (a_l == a.label && b_l == b.label)]
end

function _numlabels(::Type{A4Object}, i, j)
    return Ncache = _get_Ncache(A4Object)
end

# Data from files
# ---------------
const artifact_path = joinpath(artifact"fusiondata", "MultiTensorKit.jl-data-v0.1.1")

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
        return y
    end
end

const Ncache = IdDict{Type{<:BimoduleSector},Array{Dict{NTuple{3,Int},Int},3}}()

function _get_Ncache(::Type{T}) where {T<:BimoduleSector}
    global Ncache
    return get!(Ncache, T) do
        return extract_Nsymbol(T)
    end
end

function TensorKitSectors.Nsymbol(a::I, b::I, c::I) where {I<:A4Object}
    # TODO: should this error or return 0?
    (a.j == b.i && a.i == c.i && b.j == c.j) ||
        throw(ArgumentError("invalid fusion channel"))
    i, j, k = a.i, a.j, b.j
    return get(_get_Ncache(I)[i, j, k], (a.label, b.label, c.label), 0)
end

const Dualcache = IdDict{Type{<:BimoduleSector},Tuple{Vector{Int64}, Matrix{Vector{Int64}}}}()

function _get_dual_cache(::Type{T}) where {T<:BimoduleSector}
    global Dualcache
    return get!(Dualcache, T) do
        return extract_dual(T)
    end
end

function extract_dual(::Type{A4Object})
    N = _get_Ncache(A4Object)
    ncats = size(N, 1)
    Is = zeros(Int, ncats)
    
    map(1:ncats) do i
        Niii = N[i, i, i]
        nobji = maximum(first, keys(N[i, i, i]))
        # want to return a leftone and rightone for each entry in multifusion cat
        # leftone/rightone needs to at least be the unit object within a fusion cat
        Is[i] = findfirst(1:nobji) do a
            get(Niii, (a, a, a), 0) == 1 || return false # I x I -> I
            for othera in 1:nobji
                get(Niii, (othera, a, othera), 0) == 1 || return false # a x I -> a
                get(Niii, (a, othera, othera), 0) == 1 || return false # I x a -> a
            end

            # check leftone
            map(1:ncats) do j
                nobjj = maximum(first, keys(N[j, j, j]))
                for b in 1:nobjj
                    get(N[i, j, j], (a, b, b), 0) == 1 || return false # I = leftone(b)
                end
            end

            # check rightone
            map(1:ncats) do k
                nobjk = maximum(first, keys(N[k, k, k]))
                for c in 1:nobjk
                    get(N[k, i, k], (c, a, c), 0) == 1 || return false # I = rightone(c)
                end
            end
            return true
        end
    end
    
    allduals = 0 .|> fill(x->Vector{Int}(), ncats, ncats) # ncats square matrix of vectors
    map(1:ncats) do i
        nobji = maximum(first, keys(N[i, i, i]))
        map(1:ncats) do j
            nobjj = maximum(first, keys(N[j, j, j]))
        # the nested vectors contain the duals of the objects in 𝒞_ij, which are in C_ji 
            Niji = N[i, j, i] # 𝒞_ij x 𝒞_ji -> C_ii
            Njij = N[j, i, j] # 𝒞_ji x 𝒞_ij -> C_jj
            for i_ob in 1:nobji, j_ob in 1:nobjj
                get(Niji, (i_ob, j_ob, Is[i]), 0) == 1 || continue # leftone(c_ij) ∈ c_ij x c_ji
                get(Njij, (j_ob, i_ob, Is[j]), 0) == 1 || continue # rightone(c_ij) ∈ c_ji x c_ij
                push!(allduals[i,j], j_ob)
            end
        end
    end
    return Is, allduals
end

function Base.one(a::BimoduleSector)
    a.i == a.j || error("don't know how to define one for modules")
    return A4Object(a.i, a.i, _get_dual_cache(typeof(a))[1][a.i])
end

function TensorKitSectors.leftone(a::BimoduleSector)
    return A4Object(a.i, a.i, _get_dual_cache(typeof(a))[1][a.i])
end

function TensorKitSectors.rightone(a::BimoduleSector)
    return A4Object(a.j, a.j, _get_dual_cache(typeof(a))[1][a.j])
end

function Base.conj(a::BimoduleSector)
    return A4Object(a.j, a.i, _get_dual_cache(typeof(a))[2][a.i,a.j][a.label])
end

# function extract_Fsymbol(::Type{A4Object})
#     return mapreduce((x, y) -> cat(x, y; dims=1), 1:12) do i
#         filename = joinpath(artifact_path, "A4", "Fsymbol_$i.json")
#         @assert isfile(filename) "cannot find $filename"
#         json_string = read(filename, String)
#         Farray_part = copy(JSON3.read(json_string)) # Dict with one entry, that entry has as value a Dict (a,b,c,d,e,f) of Dicts (re, im)
#         return map(enumerate(Farray_part[Symbol(i)])) do (I, x)
#             j, k, l = Tuple(CartesianIndices((12, 12, 12))[I])
#             y = Dict{NTuple{6,Int},Array{ComplexF64,4}}()
#             for (key, v) in x
#                 a, b, c, d, e, f = parse.(Int, split(string(key)[2:(end - 1)], ", "))
#                 a_ob, b_ob, c_ob, d_ob, e_ob, f_ob = A4Object.(((i, j, a), (j, k, b),
#                                                                 (k, l, c), (i, l, d),
#                                                                 (i, k, e), (j, l, f)))
#                 result = Array{ComplexF64,4}(undef,
#                                              (Nsymbol(a_ob, b_ob, e_ob),
#                                               Nsymbol(e_ob, c_ob, d_ob),
#                                               Nsymbol(b_ob, c_ob, f_ob),
#                                               Nsymbol(a_ob, f_ob, d_ob)))
#                 #@show size(result), a_ob, b_ob, c_ob, d_ob, e_ob, f_ob, v
#                 #@show result, v
#                 s1 = length(result)
#                 s2 = length(v)
#                 @assert s1 == s2 "$s1, $s2, $a_ob, $b_ob, $c_ob, $d_ob, $e_ob, $f_ob, $v, $result"
#                 map!(result, reshape(v, size(result))) do cmplxdict
#                     return complex(cmplxdict[:re], cmplxdict[:im])
#                 end

#                 y[(a, b, c, d, e, f)] = result
#             end
#         end
#     end
# end
#TODO: figure out missing Fsymbol
function extract_Fsymbol(::Type{A4Object})
    return mapreduce((colordict, Fdict) -> cat(colordict, Fdict; dims=1), 1:12) do i
        filename = joinpath(artifact_path, "A4", "Fsymbol_$i.txt")
        @assert isfile(filename) "cannot find $filename"
        txt_string = read(filename, String)
        Farray_part = copy(readdlm(IOBuffer(txt_string))); # now a matrix with 16 columns
        Farray_part = convert_Fs(Farray_part)
        return for (colors, colordict) in Farray_part
            i,j,k,l = colors
            Fdict = Dict{NTuple{6,Int},Array{ComplexF64,4}}()
            for (labels, Fvals) in colordict
                a, b, c, d, e, f = labels
                a_ob, b_ob, c_ob, d_ob, e_ob, f_ob = A4Object.(((i, j, a), (j, k, b),
                                                                (k, l, c), (i, l, d),
                                                                (i, k, e), (j, l, f)))
                result = Array{ComplexF64,4}(undef,
                                             (Nsymbol(a_ob, b_ob, e_ob),
                                              Nsymbol(e_ob, c_ob, d_ob),
                                              Nsymbol(b_ob, c_ob, f_ob),
                                              Nsymbol(a_ob, f_ob, d_ob)))

                if length(result) == length(Fvals)                             
                    map!(result, reshape(Fvals, size(result))) do pair
                        return pair[2]
                    end
                else # due to sparse data, some Fvals are missing and we need to fill in zeros
                    # given result, can we track the missing CartesianIndices?
                    println("Mismatch in sizes: $s1, $s2, $a_ob, $b_ob, $c_ob, $d_ob, $e_ob, $f_ob")
                end

                Fdict[(a, b, c, d, e, f)] = result
            end
        end
    end
end

function convert_Fs(Farray_part::Matrix{Float64}) # Farray_part is a matrix with 16 columns
    data_dict = Dict{NTuple{4, Int}, Dict{NTuple{6, Int}, Vector{Pair{CartesianIndex{4}, ComplexF64}}}}()
    # want to make a Dict with keys (i,j,k,l) and vals 
    # a Dict with keys (a,b,c,d,e,f) and vals 
    # a pair of (mu, nu, rho, sigma) and the F value
    for row in eachrow(Farray_part)
        row = string.(split(string(row)[2:(end - 1)], ", "))
        i, j, k, l, a, b, c, d, e, f, mu, nu, rho, sigma = Int.(parse.(Float64, row[1:14]))
        v = complex(parse.(Float64, row[15:16])...)
        colordict = get!(data_dict, (i,j,k,l), Dict{NTuple{6,Int}, Vector{Pair{CartesianIndex{4}, ComplexF64}}}())
        Fdict = get!(colordict, (a, b, c, d, e, f), Vector{Pair{CartesianIndex{4}, ComplexF64}}())
        push!(Fdict, CartesianIndex(mu, nu, rho, sigma) => v)
    end
    return data_dict
end

const Fcache = IdDict{Type{<:BimoduleSector},
                      Array{Dict{NTuple{6,Int},Array{ComplexF64,4}},4}}()

function _get_Fcache(::Type{T}) where {T<:BimoduleSector}
    global Fcache
    return get!(Fcache, T) do
        return extract_Fsymbol(T)
    end
end

function TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I,
                                  f::I) where {I<:A4Object}
    # TODO: should this error or return 0?
    (a.j == b.i && b.j == c.i && a.i == d.i && c.j == d.j &&
     a.i == e.i && b.j == e.j && f.i == a.j && f.j == c.j) ||
        throw(ArgumentError("invalid fusion channel"))

    i, j, k, l = a.i, a.j, b.j, c.j
    return get(_get_Fcache(I)[i, j, k, l],
               (a.label, b.label, c.label, d.label, e.label, f.label)) do
        return zeros(sectorscalartype(A4Object),
                     (Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f),
                      Nsymbol(a, f, d)))
    end
end
