"""
    In this file we define the activation function sturcts
"""

abstract type Activation end

struct Sigmoid <: Activation end
(::Sigmoid)(z) = 1 ./ (1 .+ (exp.(-z)))
(gradient)(s::Sigmoid,z) = s(z) .* (1 .- s(z))