"""
    This file holds the definitions be it structs or abstracts for the tree models
"""

# tree structs are immutable and the value doesnt change since creation 
abstract type TreeComponent end
abstract type TreeType end
abstract type TreeModelType <: ClassicModel end

"""
    A leaf is the final node in a tree that holds a prediction
"""
struct Leaf{L} <: TreeComponent
    # the value fo the edge node (final answer based on path)
    prediction::L
end

"""
    This struct holds a branch `intermediate node` in a decision tree
    `feature`: the number of feathre the node threshold depends on
    `threshold`: the threshold is the conditions value
    `left`: left refers to another branch or leaf that has a path of (<= condition)
    `right`: right refers to another branch or leaf that has a path of (> condition)
"""
struct Branch{L} <: TreeComponent
    # the feature numebr to be referenced
    feature::Int

    # the value to be compared to
    threshold::Float64

    # the node to the left if the operation is true
    left::Union{Leaf{L},Branch{L}}

    # the node to the right if the comparison is false
    right::Union{Leaf{L},Branch{L}}
end

"""
    This struct holds the metadata for a `DecisionTreeClassifier`
    `max_depth`: the max depth that a tree can have
    `min_sample_split`: the number of samples that pass through the conditions to make it justify a valid split
    `root`: the root `head` of the tree
    `fitted`: a bool to check if the model is trained and ready to use or not
"""
struct DecisionTreeClassifier <: TreeModelType
    # max node depth the tree is allowed to build
    max_depth::Int

    # used to prevent overfitting as we are 
    min_sample_split::Int

    # the start of the tree
    root::Union{Nothing, Leaf, Branch}

    # is the model fitted or not so we can restrict or allow predictions on this instance
    fitted::Bool
end