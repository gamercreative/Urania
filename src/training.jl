"""
    Training will be held as several generalized functions and they are build as needed

    ClassicFit: this one uses hand written loss functions and is set to be used for things that are not iterative and one shot]
    DeepFit: This one will use the autograder and dynamically calculate the gradients
"""
include("model.jl")
include("loss.jl")


# @todo study for later to check if 
function ClassicFit(model::ClassicModel, ) 
end