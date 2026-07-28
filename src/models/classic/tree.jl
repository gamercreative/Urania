# ===
# Tree models
# ===

# tree structs are immutable and the value doesnt change since creation 

# a leaf is the final node in a tree that holds a prediction
struct Leaf
    # the value fo the edge node (final answer based on path)
    prediction
end

"""
    This struct holds a branch `intermediate node` in a decision tree
    `feature`: the number of feathre the node threshold depends on
    `threshold`: the threshold is the conditions value
    `left`: left refers to another branch or leaf that has a path of (<= condition)
    `right`: right refers to another branch or leaf that has a path of (> condition)
"""
struct Branch
    # the feature numebr to be referenced
    feature::Int

    # the value to be compared to
    threshold::Float64

    # the node to the left if the operation is true
    left::Union{Leaf,Branch}

    # the node to the right if the comparison is false
    right::Union{Leaf,Branch}
end

"""
    This struct holds the metadata for a `DecisionTreeClassifier`
    `max_depth`: the max depth that a tree can have
    `min_sample_split`: the number of samples that pass through the conditions to make it justify a valid split
    `root`: the root `head` of the tree
    `fitted`: a bool to check if the model is trained and ready to use or not
"""
struct DecisionTreeClassifier # <: ClassicModel
    # max node depth the tree is allowed to build
    max_depth::Int

    # used to prevent overfitting as we are 
    min_sample_split::Int

    # the start of the tree
    root::Union{Nothing, Leaf, Branch}

    # is the model fitted or not so we can restrict or allow predictions on this instance
    fitted::Bool
end

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

# the gini funciton is used to measure impurity
function gini(y::AbstractVector)
    # save legnth to save compute
    n = length(y)

    # edge case for empty arrays
    n == 0 && return 0.0

    # sort first hten use sliding window to check this is a nlog(n) approach
    s = sort(y)
    acc = 0.0
    run = 1

    # get the fraction and then 
    for i in 2:n
        if s[i] == s[i-1]
            run += 1
        else
            acc += (run / n) ^ 2
            run = 1
        end
    end

    # calculate the last run as it it not executed by the else
    acc += (run / n ) ^ 2

    # return the gini impurity calculation results
    return 1.0 - acc

end

# a function that returns the most common label
function majority(y::AbstractVector)
    # creat a new dictionary where its label -> label_count
    counts = Dict{eltype(y) , Int}()
    for entry in y
        counts[entry] = get(counts, entry, 0) + 1
    end

    # use argmax to get the label of the max count
    return argmax(counts)
end

# a function that gets the purity of a split
function split_score(x::T, y::AbstractVector, threshold::B) where {B<: Number , T <: AbstractVector{B}}
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
    return (n1/n) * gini(y_left) + (n2/n) * gini(y_right)
end

# calculate midpoints from vector
function calculate_endpoints(x::AbstractVector{<:Number})
    # get unique sorted inputs so that we can get midpoints without any duplicates its cleaner and better
    v = sort(unique(x))
    n = length(v)

    # return nothing if the feature space is constant
    n < 2 && return Float64[]

    # return the midponts of each one with x1 + x2 / 2
    return [(v[i-1] + v[i])/2 for i in 2:lastindex(v)]
end

# calculates the best split
function best_split(X::AbstractMatrix{<:Number}, y::AbstractVector)
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
            s = split_score(col, y, t)

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
function build_tree(X::AbstractMatrix{<:Real}, y::AbstractVector, max_depth::Int, min_sample_split::Int, depth::Int = 0)
    node_gini = gini(y)

    # first check conditions as this function is recurrisve
    if depth >= max_depth || length(y) < min_sample_split || node_gini == 0.0
        return Leaf(majority(y))
    end

    # get best split
    result = best_split(X, y)

    # if nothing remianed = reached the end then its time to create the end node ( leaf )
    if isnothing(result)
        return Leaf(majority(y))
    end

    # unpack the best split
    best_t, best_f, best_score = result

    # check if the score is legit better then we create the leaf
    if best_score >= node_gini
        return Leaf(majority(y))
    end

    # build it reccursively
    mask = X[:, best_f] .<= best_t
    left = build_tree(X[mask, :], y[mask], max_depth, min_sample_split, depth+1)
    right = build_tree(X[.!mask, :], y[.!mask], max_depth, min_sample_split, depth+1)

    return Branch(best_f, best_t, left, right)
end

"""
    Here is the random forest section of my file
"""