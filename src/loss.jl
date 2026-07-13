"""
    This file is used to store the loss functions that are used for classic ml mostly
    Deep learning or iterative learning will depend on autograder

    The Defined loss functions aer structs where there is the main function and there is the gradient function
    - no need to write the equations and such in teh comments as the equations are in the code themselves
    - loss functions are written to support scalar and multi-dimensional shapes
"""

abstract type Loss end

# defined loss structs start here with their respectie functions
"""
    MSE (Mean Squared Error)
"""
struct MSE <: Loss end
(::MSE)(y, ŷ) = mean(ŷ .- y) .^ 2
(gradient)(::MSE, y, ŷ) = 2 .* (ŷ .- y) ./ length(y)
