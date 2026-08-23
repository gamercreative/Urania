"""
    This file holds the definitions be it structs or abstracts for the forest models
"""

abstract type ForestModelType <: ClassicModel end

# this is the struct for a random forest of whatever tree type exists
struct RandomForest <: ForestModelType
    # a vector of tree model types
    trees::Vector{TreeModelType}

    # max depth for the tree
    max_depth

    # minimum sample split to prevent overfitting
    min_smaple_split
end
