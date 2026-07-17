# ===
# Linear models
# ===

"""
    `Linear regression` with equation y = aX + b

    `inputs`
    a: a input matrix of shape (number_of_features, batch_size)

    `model paramaters`
    X: a learnable matrix with shape (number_of_features,1)
    b: a learnable scalar
"""
mutable struct LinearRegression{B <: Number, W <:Matrix{B}} <: ClassicModel
    Weights::W
    bias::B
end