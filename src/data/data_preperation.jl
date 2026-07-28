"""
    Always provide the data as an array of (batch_size,...) if it has a batch size which is size(X,1)
"""
function bootstrap_sampling(X::AbstractArray, y::AbstractArray, n_samples::Int = size(X,1))
    batch_size = size(X,1)
    @assert batch_size == size(y,1)

    rows = rand(1:batch_size, n_samples)
    return selectdim(X, 1, rows), selectdim(y, 1, rows)
end