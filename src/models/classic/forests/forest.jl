include("definitions.jl")

#= ===
    Forest outgoing functions following the multiple dispatch principle
    holds the includes required for the Forest module to operate
=== =# 

function create_random_forest(X::AbstractMatrix{<:Real}, y::AbstractVector, tree_count::Int, max_depth::Int, min_sample_split::Int)
    # used to store the tree types
    rf = TreeModel[]

    # create the trees one by one and store them
    for _ in 1:tree_count
        X_t, y_t = bootstrap_sampling(X,y)
        push!(rf, build_classifier_tree_model(X_t, y_t, max_depth, min_sample_split))
    end

    # write the tree spec to the trees inside the forest
    specs = TreeSpecifications(
        max_depth,
        min_sample_split
    )

    # create the random forest and return
    model = RandomForest(
        specs,
        rf
    )

    return model
end

function traverse_forest(forest::ForestModel, x::AbstractVector{<:Real})
    # traverse each tree and get the prediction
    predictions = [traverse_tree(tree.root, x) for tree in forest.trees] 

    # return the most frequent prediction
    return mode(predictions)
    
end