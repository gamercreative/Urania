include("definitions.jl")

#= ===
    Forest outgoing functions following the multiple dispatch principle
    holds the includes required for the Forest module to operate
=== =# 

function create_random_forest(X::AbstractMatrix{<:Real}, y::AbstractVector, tree_count::Int, max_depth::Int, min_sample_split::Int)
    # used to store the tree types
    rf = TreeModelType[]

    # create the trees one by one and store them
    for _ in 1:tree_count
        X_t, y_t = bootstrap_sampling(X,y)
        push!(rf, build_classifier_tree_model(X_t, y_t, max_depth, min_sample_split))
    end

    # create the random forest and return
    model = RandomForest(
        rf,
        max_depth,
        min_sample_split
    )

    return model
end

function traverse_forest(forest::ForestModelType, x::AbstractMatrix{<:Real})
    
end