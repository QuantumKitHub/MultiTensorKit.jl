using TensorKitSectors
using .MultiTensorKit
using Revise

testobj = A4Object(1,1,1) # fusion cat object
unit = one(testobj)
collect(testobj⊗unit)
@assert unit == leftone(testobj) == rightone(testobj)

testobj2 = A4Object(2,2,1)
unit2 = one(testobj2)
collect(testobj2⊗unit2)
@assert unit2 == leftone(testobj2) == rightone(testobj2)

testmodobj = A4Object(1,2,1)
one(testmodobj)
leftone(testmodobj)
rightone(testmodobj)

Fsymbol(testobj, testobj, testobj, testobj, testobj, testobj)

using Artifacts
using DelimitedFiles

artifact_path = joinpath(artifact"fusiondata", "MultiTensorKit.jl-data-v0.1.1")
filename = joinpath(artifact_path, "A4", "Fsymbol_4.txt")
txt_string = read(filename, String)
F_arraypart = copy(readdlm(IOBuffer(txt_string)));

F_arraypart = MTK.convert_Fs(F_arraypart)
for (color, colordict) in F_arraypart
    for (labels, F) in colordict
        if length(F) == 2
            println(color, labels, F)
        end
    end
end
i,j,k,l = (4,12,12,2) # 5,2,8,10
a,b,c,d,e,f = (1, 11, 3, 1, 1, 3) #(2,1,1,2,2,3)
testF = F_arraypart[(i,j,k,l)][(a,b,c,d,e,f)]
a_ob, b_ob, c_ob, d_ob, e_ob, f_ob = A4Object.(((i, j, a), (j, k, b),
                                                (k, l, c), (i, l, d),
                                                (i, k, e), (j, l, f)))
result = Array{ComplexF64,4}(undef,
                            (Nsymbol(a_ob, b_ob, e_ob),
                            Nsymbol(e_ob, c_ob, d_ob),
                            Nsymbol(b_ob, c_ob, f_ob),
                            Nsymbol(a_ob, f_ob, d_ob)))

map!(result, reshape(testF, size(result))) do pair
    return pair[2]
end

function bla()
    return for (k,v) in F_arraypart
        @show k
    end
end

using JSON3
filename2 = joinpath(artifact_path, "A4", "Fsymbol_1.json")
json_string = read(filename2, String)
Farray_part2 = copy(JSON3.read(json_string));