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
