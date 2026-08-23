using Test

"""
    In this section we test data
"""
function TestBootstrapSampling()
    println("before")
    println(x_xor, y_xor)

    println("after")
    println(bootstrap_sampling(x_xor, y_xor))
end

TestBootstrapSampling()