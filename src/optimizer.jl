"""
    This file defines the optimzers
    They all share the same interface but may later break up due to more features or build ontop

    as some optimzers hold states such as SGD and Adam then they are sturcts
"""

abstract type Optimzer end

struct GD <: Optimzer
    η::Float64
end