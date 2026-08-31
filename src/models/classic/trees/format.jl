include("definitions.jl")
include("regression.jl")
include("classification.jl")
include("tree.jl")

"""
    This file holds the functions that are going to go outside the module
    such as predict and fit
"""


function fit(specs::TreeSpecifications, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    tree = build_decision_tree_model(X, y, specs.type ,specs.max_depth, specs.min_smaple_split)

    return tree
end

function predict(tree::TreeModel, X::AbstractVector{<:Real})
    return traverse_tree(tree.root, X)
end
