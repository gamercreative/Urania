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
include("models/classic/linear_regressions/linear.jl")
include("models/classic/trees/tree.jl")
include("models/classic/forests/forest.jl")

# datasets used to test across the library
include("tests/datasets.jl")

# uncomment to run tests
# include("tests/models.jl")
include("tests/data.jl")

end