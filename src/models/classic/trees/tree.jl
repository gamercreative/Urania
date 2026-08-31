#= ===
    Tree outgoing functions following the multiple dispatch principle
    holds the includes required for the Tree module to operate
=== =#

"""
    here we will define the tree impurity functions based on tree type
"""
impurity(::ClassificationTree, y) = gini(y)

"""
    here we will define the leaf value retrieval function based on tree type
"""
leaf_value(::ClassificationTree, y) = majority(y)
leaf_value(::RegressionTree, y) = mean(y)

# this is used to get next node depending on the threshold of the current node
function traverse_tree(root::Union{Leaf, Branch}, x::AbstractVector)
    # the had is either a leaf or a branch
    head::Union{Leaf, Branch} = root

    # iterate over all the branches till we stop having a branch as the next branch
    while isa(head, Branch)
        head = x[head.feature] <= head.threshold ? head.left : head.right
    end

    # if the last node we reached is a leaf then return its prediciton if not then return error
    return isa(head, Leaf) ? head.prediction : error("error the final node reached was not a leaf")
end

# a function that gets the purity of a split
function split_score(x::AbstractVector{<:Number}, y::AbstractVector, type::TreeType, threshold::Number)
    # first get hte split based on threshold
    mask = x .<= threshold 

    # split the right and left side based on the mask created
    y_left = y[mask]
    y_right = y[.!mask]

    # get the lengths for the score calculations
    n = length(y)
    n1 = length(y_left)
    n2 = length(y_right)

    # return the split purity score
    return (n1/n) * impurity(type, y_left) + (n2/n) * impurity(type, y_right)
end

# calculate midpoints from vector
function calculate_endpoints(x::AbstractVector{<:Number})
    # get unique sorted inputs so that we can get midpoints without any duplicates its cleaner and better
    v = sort(unique(x))
    n = length(v)

    # return empty vector if the feature space is constant
    n < 2 && return Float64[]

    # return the midponts of each one with x1 + x2 / 2
    return [(v[i-1] + v[i])/2 for i in 2:lastindex(v)]
end

# calculates the best split
function best_split(X::AbstractMatrix{<:Number}, y::AbstractVector, type::TreeType)
    # lets seed so the first oiption always wins the lower the better
    best_t = 0.0
    best_score = Inf
    best_f = 0

    for j in 1:size(X, 2) # type: nothing
        # here column is just the feature space
        col = X[:, j]
            

        # loop over the thresholds and check purity
        for t in calculate_endpoints(col)
            
            # get the split purity score
            s = split_score(col, y, type, t)

            # if the score is smaller then the purity is lower than pick it as the best
            if s < best_score
                best_f = j
                best_t = t
                best_score = s
            end

        end
    end

    best_f == 0 && return nothing
    return (best_t, best_f, best_score)
end

# function to build the tree
function build_decision_tree_node(X::AbstractMatrix{<:Real}, y::AbstractVector, type::TreeType, max_depth::Int, min_sample_split::Int, depth::Int = 0)
    node_impurity = impurity(type, y)

    # first check conditions as this function is recurrisve
    if depth >= max_depth || length(y) < min_sample_split || node_impurity == 0.0
        return Leaf(leaf_value(type,y))
    end

    # get best split
    result = best_split(X, y, type)

    # if nothing remianed = reached the end then its time to create the end node ( leaf )
    if isnothing(result)
        return Leaf(leaf_value(type,y))
    end

    # unpack the best split
    best_t, best_f, best_score = result

    # build it reccursively
    mask = X[:, best_f] .<= best_t
    left = build_decision_tree_node(X[mask, :], y[mask], type, max_depth, min_sample_split, depth+1)
    right = build_decision_tree_node(X[.!mask, :], y[.!mask], type, max_depth, min_sample_split, depth+1)

    return Branch(best_f, best_t, left, right)
end

function build_decision_tree_model(X::AbstractMatrix{<:Real}, y::AbstractVector, type::TreeType, max_depth::Int, min_sample_split::Int)
    tree = build_decision_tree_node(X, y, type, max_depth, min_sample_split)

    # the model itself is a struct with the tree and the specs of the tree
    model = DecisionTree(
        type,
        max_depth,
        min_sample_split,
        tree
    )

    # return the model
    return model
end