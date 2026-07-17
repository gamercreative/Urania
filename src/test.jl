# ===
# tree section
# ===
using Test

include("models/model.jl")
include("models/classic/tree.jl")

function TestTreeClassifier()
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
    @test traverse_tree(b1,[0.0, 0.0] ) == 0
    @test traverse_tree(b1, [0.0, 1.0]) == 1
    @test traverse_tree(b1, [1.0, 0.0]) == 1
    @test traverse_tree(b1, [1.0, 1.0]) == 0

    # end test case
    end

    @testset "auto generated xor tree test" begin

    end
end

TestTreeClassifier()