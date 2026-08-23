"""
    This file holds the definitions be it structs or abstracts for the forest models
"""

abstract type ForestModel <: ClassicModel end

# this is the struct for a random forest of whatever tree type exists
struct RandomForest <: ForestModel
    # store the tree specifications like max_depth and min_sample_split
    specs::TreeSpecifications
    
    # a vector of tree model types
    trees::Vector{TreeModel}
end

function is_fitted(forest::ForestModel)
    # if its empty not trees then its not fitted
    isempty(forest.trees) && return false

    for tree in forest.trees
        # if hte tree is not fitted then return false
        !is_fitted(tree) && return false
    end

    # return fitted if it has trees and each one is fitted
    return true
end