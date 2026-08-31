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