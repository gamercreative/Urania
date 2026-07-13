# the mother interface for all model
abstract type Model end

# the children interfaces for the mother model
"""
    used for classic machine learning models (logistical regression, trees, etc...)
    and mostly used for closed-form solution models
"""
abstract type ClassicModel <: Model end 

"""
    used for more abstract deep learning models that do not have a definite shape
    and mostly for iterative traning
"""
abstract type DeepLearningModel <: Model end

# Classic model definitions

# classic machine learning algorithms
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