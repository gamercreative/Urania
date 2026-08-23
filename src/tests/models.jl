using Test

"""
    In this section we test models
"""


x_xor = [0.0 0.0 
    0.0 1.0 
    1.0 0.0 
    1.0 1.0]

y_xor = [0.0, 1.0, 1.0, 0.0]

function TestCreateTreeClassifier()

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
    @test traverse_tree(b1, x_xor[1,:] ) == y_xor[1]
    @test traverse_tree(b1, x_xor[2,:]) == y_xor[2]
    @test traverse_tree(b1, x_xor[3,:]) == y_xor[3]
    @test traverse_tree(b1, x_xor[4,:]) == y_xor[4]

    # end test case
    end

    @testset "automatic XOR tree test" begin
        head = build_classifier_tree_node(x_xor, y_xor, 5, 2)
        dump(head)

        @test traverse_tree(head, x_xor[1,:]) == y_xor[1]
        @test traverse_tree(head, x_xor[2,:]) == y_xor[2]
        @test traverse_tree(head, x_xor[3,:]) == y_xor[3]
        @test traverse_tree(head, x_xor[4,:]) == y_xor[4]
    end

end

function TestTreeClassifier(tree::Union{Branch, Leaf})

    @testset "premade XOR classifier tree test" begin

    # traverse and inspect the outputs
    @test traverse_tree(tree, x_xor[1,:] ) == y_xor[1]
    @test traverse_tree(tree, x_xor[2,:]) == y_xor[2]
    @test traverse_tree(tree, x_xor[3,:]) == y_xor[3]
    @test traverse_tree(tree, x_xor[4,:]) == y_xor[4]

    # end test case
    end

end

function TestRandomForest()
    @testset "random forest construction" begin
        forest = create_random_forest(x_xor, y_xor, 2, 5, 2)

        for tree in forest.trees
            @test tree isa DecisionTreeClassifier
            @test tree.fitted
            @test tree.root isa Union{Leaf, Branch}
        end
    end
end

TestRandomForest()