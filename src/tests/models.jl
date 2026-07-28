using Test

include("../models/model.jl")
include("../models/classic/tree.jl")

"""
    In this section we test models
"""

function TestTreeClassifier()
    x = [0.0 0.0 
        0.0 1.0 
        1.0 0.0 
        1.0 1.0]
    y = [0.0, 1.0, 1.0, 0.0]

    # first test the structure and forward pass of the classifier tree
    @testset "handmade XOR tree test" begin
    # leafs generated
    l1 = Leaf(0)
    l2 = Leaf(1)

    # branches generated
    b1_1 = Branch(2, 0.5, l1, l2)
    b1_2 = Branch(2, 0.5, l2 , l1)
    b1 = Branch(1 ,0.5, b1_1, b1_2)

    # traverse and inspect the outputs
    @test traverse_tree(b1, x[1,:] ) == y[1]
    @test traverse_tree(b1, x[2,:]) == y[2]
    @test traverse_tree(b1, x[3,:]) == y[3]
    @test traverse_tree(b1, x[4,:]) == y[4]

    # end test case
    end

     "automatic XOR tree test" 
        head = build_tree(x, y, 2, 1)
        dump(head)

        traverse_tree(head, x[1,:]) == y[1]
        traverse_tree(head, x[2,:]) == y[2]
        traverse_tree(head, x[3,:]) == y[3]
        traverse_tree(head, x[4,:]) == y[4]

end

TestTreeClassifier()