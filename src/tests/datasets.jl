# xor gate logic mapped
x_xor = [0.0 0.0 
    0.0 1.0 
    1.0 0.0 
    1.0 1.0]

y_xor = [0.0, 1.0, 1.0, 0.0]

# random claude data
const X_hand_train = Float64[   1   2   2   7    #  1  class 0
                                2   1   1   3    #  2  class 0
                                2   7   3   9    #  3  class 0  <- x2 lies
                                3   2   2   1    #  4  class 0
                                1   4   1   5    #  5  class 0
                                3   1   4   8    #  6  class 0
                                4   8   3   2    #  7  class 0  <- x2 lies
                                2   2   2   6    #  8  class 0
                                7   6   8   4    #  9  class 1
                                8   7   7   9    # 10  class 1
                                9   3   9   2    # 11  class 1  <- x2 lies
                                6   8   6   7    # 12  class 1
                                8   6   7   1    # 13  class 1
                                7   2   9   5    # 14  class 1  <- x2 lies
                                9   8   8   8    # 15  class 1
                                6   5   6   3 ]  # 16  class 1
 
const y_hand_train = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
 
#                             x1  x2  x3  x4
const X_hand_test = Float64[   1   3   2   8    # 1  y=0  easy
                               3   6   1   2    # 2  y=0  TRAP: x2 says class 1
                               2   2   3   5    # 3  y=0  easy
                               4   1   4   9    # 4  y=0  near boundary on x1
                               8   7   8   1    # 5  y=1  easy
                               6   2   7   6    # 6  y=1  TRAP: x2 says class 0
                               9   6   9   3    # 7  y=1  easy
                               7   8   6   4 ]  # 8  y=1  near boundary on x3
 
const y_hand_test = [0, 0, 0, 0, 1, 1, 1, 1]
