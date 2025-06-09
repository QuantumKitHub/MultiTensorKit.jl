struct BimoduleSector{Name} <: Sector
    i::Int
    j::Int
    label::Int
    function BimoduleSector{:A4}(i::Int, j::Int, label::Int)
        i <= 12 && j <= 12 || throw(DomainError("object outside the matrix A4"))
        return label <= _numlabels(BimoduleSector{:A4}, i, j) ? new{:A4}(i, j, label) :
               throw(DomainError("label outside category A4($i, $j)"))
    end
end
BimoduleSector{Name}(data::NTuple{3,Int}) where {Name} = BimoduleSector{Name}(data...)
const A4Object = BimoduleSector{:A4}

# Utility implementations
# -----------------------
function Base.isless(a::I, b::I) where {I<:BimoduleSector}
    return isless((a.i, a.j, a.label), (b.i, b.j, b.label))
end
Base.hash(a::BimoduleSector, h::UInt) = hash(a.i, hash(a.j, hash(a.label, h)))
function Base.convert(::Type{BimoduleSector{Name}}, d::NTuple{3,Int}) where {Name}
    return BimoduleSector{Name}(d...)
end

Base.IteratorSize(::Type{<:SectorValues{<:BimoduleSector}}) = Base.SizeUnknown()

# TODO: generalize?
function Base.iterate(iter::SectorValues{A4Object}, (I, label)=(1, 1))
    I > 12 * 12 && return nothing
    i, j = CartesianIndices((12, 12))[I].I
    maxlabel = _numlabels(A4Object, i, j)
    return if label > maxlabel
        iterate(iter, (I + 1, 1))
    else
        A4Object(i, j, label), (I, label + 1)
    end
end

function Base.length(::SectorValues{A4Object})
    return sum(_numlabels(A4Object, i, j) for i in 1:12, j in 1:12)
end

TensorKitSectors.FusionStyle(::Type{A4Object}) = GenericFusion()
TensorKitSectors.BraidingStyle(::Type{A4Object}) = NoBraiding()
TensorKitSectors.sectorscalartype(::Type{A4Object}) = ComplexF64

function TensorKitSectors.:⊗(a::A4Object, b::A4Object)
    @assert a.j == b.i
    Ncache = _get_Ncache(A4Object)[a.i, a.j, b.j]
    return A4Object[A4Object(a.i, b.j, c_l)
                    for (a_l, b_l, c_l) in keys(Ncache)
                    if (a_l == a.label && b_l == b.label)]
end

function _numlabels(::Type{T}, i, j) where {T<:BimoduleSector}
    return length(_get_dual_cache(T)[2][i, j])
end

# Data from files
# ---------------
const artifact_path = joinpath(artifact"fusiondata", "MultiTensorKit.jl-data-v0.1.2")

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
        @debug "loading Nsymbol cache for $T"
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

const Dualcache = IdDict{Type{<:BimoduleSector},Tuple{Vector{Int64},Matrix{Vector{Int64}}}}()

function _get_dual_cache(::Type{T}) where {T<:BimoduleSector}
    global Dualcache
    return get!(Dualcache, T) do
        @debug "loading dual cache for $T"
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

    allduals = Matrix{Vector{Int}}(undef, ncats, ncats) # ncats square matrix of vectors
    for i in 1:ncats
        nobji = maximum(first, keys(N[i, i, i]))
        for j in 1:ncats
            allduals[i, j] = Int[]

            nobjj = maximum(first, keys(N[j, j, j]))
            # the nested vectors contain the duals of the objects in 𝒞_ij, which are in C_ji 
            Niji = N[i, j, i] # 𝒞_ij x 𝒞_ji -> C_ii
            Njij = N[j, i, j] # 𝒞_ji x 𝒞_ij -> C_jj
            for i_ob in 1:nobji, j_ob in 1:nobjj
                get(Niji, (i_ob, j_ob, Is[i]), 0) == 1 || continue # leftone(c_ij) ∈ c_ij x c_ji
                get(Njij, (j_ob, i_ob, Is[j]), 0) == 1 || continue # rightone(c_ij) ∈ c_ji x c_ij
                push!(allduals[i, j], j_ob)
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
    return A4Object(a.j, a.i, _get_dual_cache(typeof(a))[2][a.i, a.j][a.label])
end

function extract_Fsymbol(::Type{A4Object})
    result = Dict{NTuple{4,Int},Dict{NTuple{6,Int},Array{ComplexF64,4}}}()
    for i in 1:12
        filename = joinpath(artifact_path, "A4", "Fsymbol_$i.txt")
        @debug "loading $filename"
        @assert isfile(filename) "cannot find $filename"
        Farray_part = readdlm(filename)
        for ((i, j, k, l), colordict) in convert_Fs(Farray_part)
            result[(i, j, k, l)] = Dict{NTuple{6,Int},Array{ComplexF64,4}}()
            for ((a, b, c, d, e, f), Fvals) in colordict
                a_ob, b_ob, c_ob, d_ob, e_ob, f_ob = A4Object.(((i, j, a), (j, k, b),
                                                                (k, l, c), (i, l, d),
                                                                (i, k, e), (j, l, f)))
                result[(i, j, k, l)][(a, b, c, d, e, f)] = zeros(ComplexF64,
                                                                 Nsymbol(a_ob, b_ob, e_ob),
                                                                 Nsymbol(e_ob, c_ob, d_ob),
                                                                 Nsymbol(b_ob, c_ob, f_ob),
                                                                 Nsymbol(a_ob, f_ob, d_ob))
                for (I, v) in Fvals
                    result[(i, j, k, l)][(a, b, c, d, e, f)][I] = v
                end
            end
        end
    end
    return result
end

function convert_Fs(Farray_part::Matrix{Float64}) # Farray_part is a matrix with 16 columns
    data_dict = Dict{NTuple{4,Int},
                     Dict{NTuple{6,Int},Vector{Pair{CartesianIndex{4},ComplexF64}}}}()
    # want to make a Dict with keys (i,j,k,l) and vals 
    # a Dict with keys (a,b,c,d,e,f) and vals 
    # a pair of (mu, nu, rho, sigma) and the F value
    for row in eachrow(Farray_part)
        i, j, k, l, a, b, c, d, e, f, mu, nu, rho, sigma = Int.(@view(row[1:14]))
        v = complex(row[15], row[16])
        colordict = get!(data_dict, (i, j, k, l),
                         Dict{NTuple{6,Int},Vector{Pair{CartesianIndex{4},ComplexF64}}}())
        Fdict = get!(colordict, (a, b, c, d, e, f),
                     Vector{Pair{CartesianIndex{4},ComplexF64}}())
        push!(Fdict, CartesianIndex(mu, nu, rho, sigma) => v)
    end
    return data_dict
end

const Fcache = IdDict{Type{<:BimoduleSector},
                      Dict{NTuple{4,Int64},Dict{NTuple{6,Int64},Array{ComplexF64,4}}}}()

function _get_Fcache(::Type{T}) where {T<:BimoduleSector}
    global Fcache
    return get!(Fcache, T) do
        @debug "loading Fsymbol cache for $T"
        return extract_Fsymbol(T)
    end
end

function TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I,
                                  f::I) where {I<:A4Object}
    # required to keep track of multiplicities where F-move is partially unallowed
    # also deals with invalid fusion channels
    Nabe = Nsymbol(a, b, e)
    Necd = Nsymbol(e, c, d)
    Nbcf = Nsymbol(b, c, f)
    Nafd = Nsymbol(a, f, d)

    Nabe > 0 && Necd > 0 && Nbcf > 0 && Nafd > 0 ||
        return zeros(sectorscalartype(I), Nabe, Necd, Nbcf, Nafd)

    i, j, k, l = a.i, a.j, b.j, c.j
    colordict = _get_Fcache(I)[i, j, k, l]
    return colordict[(a.label, b.label, c.label, d.label, e.label, f.label)]
end

# interface with TensorKit where necessary
#-----------------------------------------

function TensorKit.blocksectors(W::TensorMapSpace{S,N₁,N₂}) where
         {S<:Union{GradedSpace{A4Object,NTuple{486,Int64}},
                   SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}}},N₁,N₂}
    codom = codomain(W)
    dom = domain(W)
    # @info "in the correct blocksectors"
    if N₁ == 0 && N₂ == 0 # 0x0-dimensional TensorMap is just a scalar, return all units
        # this is a problem in full contractions where the coloring outside is 𝒞
        return NTuple{12,A4Object}(one(A4Object(i, i, 1)) for i in 1:12) # have to return all units b/c no info on W in this case
    elseif N₁ == 0
        @assert N₂ != 0 "one of Type A4Object doesn't exist"
        return filter!(isone, collect(blocksectors(dom)))
    elseif N₂ == 0
        @assert N₁ != 0 "one of Type A4Object doesn't exist"
        return filter!(isone, collect(blocksectors(codom)))
    elseif N₂ <= N₁ # keep intersection
        return filter!(c -> hasblock(codom, c), collect(blocksectors(dom)))
    else
        return filter!(c -> hasblock(dom, c), collect(blocksectors(codom)))
    end
end

# TODO: definition for zero of GradedSpace?

function dim(V::GradedSpace{<:BimoduleSector})
    T = Base.promote_op(*, Int, real(sectorscalartype(sectortype(V))))
    return reduce(+, dim(V, c) * dim(c) for c in sectors(V); init=zero(T))
end

# limited oneunit 
function Base.oneunit(S::GradedSpace{<:BimoduleSector})
    allequal(a.i for a in sectors(S)) && allequal(a.j for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S are not all equal"))
    first(sectors(S)).i == first(sectors(S)).j ||
        throw(ArgumentError("sectors of $S are non-diagonal"))
    sector = one(first(sectors(S)))
    return ℂ[A4Object](sector => 1)
end

function Base.oneunit(S::SumSpace{<:GradedSpace{<:BimoduleSector}})
    allequal(a.i for a in sectors(S)) && allequal(a.j for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S are not all equal"))
    first(sectors(S)).i == first(sectors(S)).j ||
        throw(ArgumentError("sectors of $S are non-diagonal"))
    sector = one(first(sectors(S)))
    return SumSpace(ℂ[A4Object](sector => 1))
end

# maybe from the homspace
function TensorKit.insertrightunit(P::ProductSpace{V,N}, ::Val{i}=Val(length(P));
                                   conj::Bool=false,
                                   dual::Bool=false) where {i,V<:GradedSpace{<:I},N} where {I<:BimoduleSector}
    #possible change to rightone of correct space for N = 0
    u = N > 0 ? oneunit(P[1]) : error("no unit object in this space")
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i, (u,)))
end

function TensorKit.insertleftunit(P::ProductSpace{V,N}, ::Val{i}=Val(length(P) + 1);
                                  conj::Bool=false,
                                  dual::Bool=false) where {i,V<:GradedSpace{<:I},N} where {I<:BimoduleSector}
    u = N > 0 ? oneunit(P[1]) : error("no unit object in this space")
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i - 1, (u,)))
end
