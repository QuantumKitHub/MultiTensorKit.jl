struct BimoduleSector{Name} <: Sector
    i::Int
    j::Int
    label::Int
    function BimoduleSector{:A4}(i::Int, j::Int, label::Int)
        i <= size(BimoduleSector{:A4}) && j <= size(BimoduleSector{:A4}) ||
            throw(DomainError("object outside the matrix A4"))
        return label <= _numlabels(BimoduleSector{:A4}, i, j) ? new{:A4}(i, j, label) :
               throw(DomainError("label outside category A4($i, $j)"))
    end
end
BimoduleSector{Name}(data::NTuple{3,Int}) where {Name} = BimoduleSector{Name}(data...)
const A4Object = BimoduleSector{:A4}

Base.convert(::Type{<:BimoduleSector{Name}}, labels::NTuple{3,Int}) where {Name} = BimoduleSector{Name}(labels...)

function Base.show(io::IO, a::BimoduleSector{Name}) where {Name}
    if get(io, :typeinfo, nothing) === typeof(a)
        print(io, (a.i, a.j, a.label))
    else
        print(io, typeof(a), (a.i, a.j, a.label))
    end
    return nothing
end

# Utility implementations
# -----------------------
function Base.isless(a::I, b::I) where {I<:BimoduleSector}
    return isless((a.i, a.j, a.label), (b.i, b.j, b.label))
end
Base.hash(a::BimoduleSector, h::UInt) = hash(a.i, hash(a.j, hash(a.label, h)))
function Base.convert(::Type{BimoduleSector{Name}}, d::NTuple{3,Int}) where {Name}
    return BimoduleSector{Name}(d...)
end

Base.size(::Type{A4Object}) = 7

Base.IteratorSize(::Type{<:SectorValues{<:BimoduleSector}}) = Base.SizeUnknown()

# TODO: generalize?
function Base.iterate(iter::SectorValues{A4Object}, (I, label)=(1, 1))
    s = size(A4Object)
    I > s * s && return nothing
    i, j = CartesianIndices((s, s))[I].I
    maxlabel = _numlabels(A4Object, i, j)
    return if label > maxlabel
        iterate(iter, (I + 1, 1))
    else
        A4Object(i, j, label), (I, label + 1)
    end
end

function Base.length(::SectorValues{A4Object})
    s = size(A4Object)
    return sum(_numlabels(A4Object, i, j) for i in 1:s, j in 1:s)
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

# User-friendly functions
# -------------------
#TODO: add functions to identify categories

# Data from files
# ---------------
const artifact_path = joinpath(artifact"fusiondata", "MultiTensorKit.jl-data-v0.1.5")

function extract_Nsymbol(::Type{A4Object})
    filename = joinpath(artifact_path, "A4", "Nsymbol.txt")
    isfile(filename) || throw(LoadError(filename, 0, "Nsymbol file not found for A4"))
    Narray = readdlm(filename) # matrix with 7 columns

    data_dict = Dict{NTuple{3,Int},Dict{NTuple{3,Int},Int}}()
    for row in eachrow(Narray)
        i, j, k, a, b, c, N = Int.(@view(row[1:size(A4Object)]))
        colordict = get!(data_dict, (i, j, k), Dict{NTuple{3,Int},Int}())
        push!(colordict, (a, b, c) => N)
    end

    return data_dict
end

const Ncache = IdDict{Type{<:BimoduleSector},
                      Dict{NTuple{3,Int},Dict{NTuple{3,Int},Int}}}()

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

const Dualcache = IdDict{Type{<:BimoduleSector},Tuple{Vector{Int64},Matrix{Vector{Int64}}}}()

function _get_dual_cache(::Type{T}) where {T<:BimoduleSector}
    global Dualcache
    return get!(Dualcache, T) do
        return extract_dual(T)
    end
end

function extract_dual(::Type{A4Object})
    N = _get_Ncache(A4Object)
    ncats = size(A4Object)
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
    a.i == a.j || throw(DomainError("unit of module category ($(a.i), $(a.j)) of $(typeof(a)) is ill-defined"))
    return typeof(a)(a.i, a.i, _get_dual_cache(typeof(a))[1][a.i])
end

Base.isone(a::BimoduleSector) = leftone(a) == a == rightone(a)

function Base.one(::Type{<:BimoduleSector})
    throw(ArgumentError("one of Type BimoduleSector doesn't exist"))
end

function TensorKitSectors.leftone(a::BimoduleSector)
    return typeof(a)(a.i, a.i, _get_dual_cache(typeof(a))[1][a.i])
end

function TensorKitSectors.rightone(a::BimoduleSector)
    return typeof(a)(a.j, a.j, _get_dual_cache(typeof(a))[1][a.j])
end

function Base.conj(a::BimoduleSector)
    return typeof(a)(a.j, a.i, _get_dual_cache(typeof(a))[2][a.i, a.j][a.label])
end

function extract_Fsymbol(::Type{A4Object})
    result = Dict{NTuple{4,Int},Dict{NTuple{6,Int},Array{ComplexF64,4}}}()
    filename = joinpath(artifact_path, "A4", "Fsymbol.txt")
    @assert isfile(filename) "cannot find $filename"
    Farray = readdlm(filename)
    for ((i, j, k, l), colordict) in convert_Fs(Farray)
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

    zero_array = zeros(sectorscalartype(I), Nabe, Necd, Nbcf, Nafd)
    Nabe > 0 && Necd > 0 && Nbcf > 0 && Nafd > 0 ||
        return zero_array

    i, j, k, l = a.i, a.j, b.j, c.j
    colordict = _get_Fcache(I)[i, j, k, l]
    return get!(colordict, (a.label, b.label, c.label, d.label, e.label, f.label), zero_array)
end

# interface with TensorKit where necessary
#-----------------------------------------

function TensorKit.blocksectors(W::TensorMapSpace{S,N₁,N₂}) where
         {S<:Union{Vect[A4Object],
                   SumSpace{Vect[A4Object]}},N₁,N₂}
    codom = codomain(W)
    dom = domain(W)
    if N₁ == 0 && N₂ == 0 # 0x0-dimensional TensorMap is just a scalar, return all units
        # this is a problem in full contractions where the coloring outside is 𝒞
        return NTuple{size(A4Object),A4Object}(one(A4Object(i, i, 1))
                                               for i in 1:size(A4Object)) # have to return all units b/c no info on W in this case
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

function TensorKit.dim(V::GradedSpace{<:BimoduleSector})
    T = Base.promote_op(*, Int, real(sectorscalartype(sectortype(V))))
    return reduce(+, dim(V, c) * dim(c) for c in sectors(V); init=zero(T))
end

Base.zero(S::Type{<:GradedSpace{<:BimoduleSector}}) = S()

function TensorKit.fuse(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:BimoduleSector}
    dims = TensorKit.SectorDict{I,Int}()
    for a in sectors(V₁), b in sectors(V₂)
        a.j == b.i || continue # skip if not compatible
        for c in a ⊗ b
            dims[c] = get(dims, c, 0) + Nsymbol(a, b, c) * dim(V₁, a) * dim(V₂, b)
        end
    end
    return typeof(V₁)(dims)
end

# limited oneunit 
function Base.oneunit(S::GradedSpace{<:BimoduleSector})
    allequal(a.i for a in sectors(S)) && allequal(a.j for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S are not all equal"))
    first(sectors(S)).i == first(sectors(S)).j ||
        throw(ArgumentError("sectors of $S are non-diagonal"))
    sector = one(first(sectors(S)))
    return spacetype(S)(sector => 1)
end

function Base.oneunit(S::SumSpace{<:GradedSpace{<:BimoduleSector}})
    @assert !isempty(S) "Cannot determine type of empty space"
    return SumSpace(oneunit(first(S.spaces)))
end

# oneunit for spaces whose elements all belong to the same sector
function TensorKit.rightoneunit(S::GradedSpace{<:BimoduleSector})
    allequal(a.j for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S do not have the same rightone"))

    sector = rightone(first(sectors(S)))
    return spacetype(S)(sector => 1)
end

function TensorKit.rightoneunit(S::SumSpace{<:GradedSpace{<:BimoduleSector}})
    @assert !isempty(S) "Cannot determine type of empty space"
    return SumSpace(rightoneunit(first(S.spaces)))
end

function TensorKit.leftoneunit(S::GradedSpace{<:BimoduleSector})
    allequal(a.i for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S do not have the same leftone"))

    sector = leftone(first(sectors(S)))
    return spacetype(S)(sector => 1)
end

function TensorKit.leftoneunit(S::SumSpace{<:GradedSpace{<:BimoduleSector}})
    @assert !isempty(S) "Cannot determine type of empty space"
    return SumSpace(leftoneunit(first(S.spaces)))
end

function TensorKit.insertrightunit(P::ProductSpace{V,N}, ::Val{i};
                                   conj::Bool=false,
                                   dual::Bool=false) where {i,V<:GradedSpace{I},N} where {I<:BimoduleSector}
    i > N && error("cannot insert a sensible right unit onto $P at index $(i+1)")
    # possible change to rightone of correct space for N = 0
    u = N > 0 ? rightoneunit(P[i]) : error("no unit object in $P")
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i, (u,)))
end

# possible TODO: overwrite defaults at level of HomSpace and TensorMap?
function TensorKit.insertleftunit(P::ProductSpace{V,N}, ::Val{i}; # want no defaults?
                                  conj::Bool=false,
                                  dual::Bool=false) where {i,V<:GradedSpace{I},N} where {I<:BimoduleSector}
    i > N && error("cannot insert a sensible left unit onto $P at index $i") # do we want this to error in the diagonal case?
    u = N > 0 ? leftoneunit(P[i]) : error("no unit object in $P")
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i - 1, (u,)))
end

function TensorKit.scalar(t::AbstractTensorMap{T,S,0,0}) where {T,
                                                                S<:GradedSpace{<:BimoduleSector}}
    Bs = collect(blocks(t))
    inds = findall(!iszero ∘ last, Bs)
    isempty(inds) && return zero(scalartype(t))
    return only(last(Bs[only(inds)]))
end

# is this even necessary? can let it error at TensorKit fusiontrees.jl:93 from the one(<:BimoduleSector) call
# but these errors are maybe more informative
function TensorKit.FusionTree(uncoupled::Tuple{<:I,Vararg{I}}) where {I<:BimoduleSector}
    coupled = collect(⊗(uncoupled...))
    if length(coupled) == 0 # illegal fusion somewhere
        throw(ArgumentError("Forbidden fusion with uncoupled sectors $uncoupled"))
    else # allowed fusions require inner lines
        error("fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`")
    end
end

# this one might also be overkill, since `FusionTreeIterator`s don't check whether the fusion is allowed
function TensorKit.fusiontrees(uncoupled::Tuple{I,Vararg{I}}) where {I<:BimoduleSector}
    return throw(ArgumentError("coupled sector must be provided for $I fusion"))
end