module MultiTensorKit

export BimoduleSector, A4Object

using DelimitedFiles
using Pkg
using Pkg.Artifacts
using TensorKitSectors

using TensorKit
import TensorKit: hasblock, dim

include("bimodulesector.jl")

end
