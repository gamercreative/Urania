"""
    this file contains the backprop algorithms
"""

include("loss.jl")
include("activation.jl")

"""
    This is a backprop algorithm built for linear models and neural networks later on
    it goes around like this
    ∂L/∂ŷ derivative of the loss
    ∂ŷ/∂w derviative of the activation
    here this is the result of the chain rule from the final output from the loss to the bare numbers
"""
function LinearBackPropagation(loss::Loss, activation::Activation, X, y, ŷ, z)
    ∂L_∂ŷ = gradient(loss, y, ŷ)
    ∂ŷ_∂z = gradient(activation, z)
    δ = ∂L_∂ŷ .* ∂ŷ_∂z
    
    ∂L_∂w = X' * δ
    ∂L_∂b = sum(δ)

    return ∂L_∂w, ∂L_∂b
end

"""
    for the linear backprop with CrossEntropy and SoftMax the result of the derivative should be a jacobian matrix
    so because of this we will create the final collapsed version
"""
# function LinearBackPropagation(loss::CrossEntropy, activation::SoftMax, X, y, ŷ, z)