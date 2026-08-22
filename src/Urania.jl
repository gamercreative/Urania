"""
    Urania

Machine learning library built from scratch in Julia.
"""
module Urania

using Statistics

# model base declarations
include("models/model.jl")

# core
include("core/activation.jl")
include("core/loss.jl")
include("core/optimizer.jl")
include("core/regularization.jl")
include("core/training.jl")

# data
include("data/data_preperation.jl")


# model-specific pipelines
include("models/classic/linear.jl")
include("models/classic/tree.jl")

end