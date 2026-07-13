"""
    This file is used to store the loss functions that are used for classic ml mostly
    Deep learning or iterative learning will depend on autograder

    The Defined loss functions aer structs where there is the main function and there is the gradient function
    no need to write the equations and such in teh comments as the equations are in the code themselves
"""

abstract type DefinedLoss end

# defined loss structs start here with their respectie functions
"""
    MSE (Mean Squared Error)
"""
struct MSE <: DefinedLoss
    (::MSE)(y, ŷ) = mean(ŷ .- y)^2
    (Gradient)(::MSE, y, ŷ) = 2 .* (ŷ .- y) ./ length(y)
end
