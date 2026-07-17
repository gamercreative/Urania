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