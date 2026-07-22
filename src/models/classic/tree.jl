# ===
# Tree models
# ===

# tree structs are immutable and the value doesnt change since creation 

# a leaf is the final node in a tree that holds a prediction
struct Leaf
    # the value fo the edge node (final answer based on path)
    prediction
end

# a branch is a node inside the tree that holds the path metadata that leads to anotehr branch or leaf
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

# a tree used to classify data
struct DecisionTreeClassifier <: ClassicModel
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
        head = x[head.Feature] <= head.threshold ? head.left : head.right
    end

    # if the last node we reached is a leaf then return its prediciton if not then return error
    return isa(head, Leaf) ? head.prediction : error("error the final node reached was not a leaf")
end

# the gini funciton is used to measure impurity
function gini(y::AbstractArray)
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
function split_score(x::T, y::T, threshold::B) where {B<: Number , T <: AbstractArray{B}}
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

function best_split()
    
end