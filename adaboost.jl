using DataFrames
using CSV

include("decisiontree.jl")

nodes = Array{Node}

function Adaboost(train, test, max_iterations = 5)
    # train x, y
    x_train = train[:,1:size(train,2)-1]
    y_train = train[:,size(train,2):size(train,2)]
    
    # test x, y
    x_test = test[:,1:size(test,2)-1]
    y_test = test[:,size(test,2):size(test,2)]

    # size of DataSet
    N = size(x_train, 1)
    
    # initialize W
    upper = 1 / max_iterations
    x     = x_train[ 1 : trunc(Int,upper * N), : ]
    W     = [ [1 / size(x,1) for i in range(1, length=size(x,1))] for j in range(1, length=max_iterations)]
    
    for iteration in range(2, length=max_iterations)
        lower = ( iteration - 1 ) / max_iterations
        upper = iteration / max_iterations
        
        x = x_train[1 + trunc(Int, lower * N ) : trunc(Int,upper * N) ,:]
        y = y_train[1 + trunc(Int, lower * N ) : trunc(Int,upper * N) ,:]
        x_y_train = train[1 + trunc(Int, lower * N ) : trunc(Int,upper * N) ,:]

        #@show  Compute_Distribuiton(x_y_train, x, y, W[iteration - 1])
        W[iteration] = Compute_Distribuiton(train, x, y, W[iteration - 1])
    end

    @show W
end

function Compute_Distribuiton(train, x_train_i, y_train_i, W_i)
    
    # weight for iteration
    W = [1/size(x_train_i,1) for i in range(1,length=size(x_train_i,1))]

    # train the x_train_i dataset
    train_distribuiton(train)
    
    # compute error
    e_i     = 0.5 - 0.5 * ( sum( ( W_i[i] * y_train_i[i] * h(x_train_i[i,:]) ) for i in range(1,length=size(x_train_i,1)) ) )
    
    # compute votes
    omega_i = 0.5 * log( (1 - e_i) / e_i )

    # update weight
    W   = [ W_i[i] * exp( ( (-1 * y_train_i[i]) * omega_i * h(x_train_i[i,:]) ) for i in range(1,length=size(x_train_i,1)) ) ]

    # normalize W
    Z_i = maximum(W)
    W   = [W[i] / Z_i for i in range(1,length=size(W,1))]

    return W
end

function train_distribuiton(train)
    nodes = decision_tree(train)
end

function h(x)
    return prediction(x, nodes)
end

function Compute_Weight(error_weight)
    return 0.5 * log((1-error_weight)/error_weight)
end

function train_test(S, test_size = 0.25)
    train_in = S[trunc(Int, floor( size(S,1) * test_size ) ) + 1 : size(S,1),:]
    test_in  = S[1:trunc(Int, floor( size(S,1) * test_size ) ),:]
    return train_in, test_in
end

# Random dataset
S_dataset   = CSV.read("dataframe.csv")

# Split the data into train_dataset and test_dataset
train, test = train_test(S_dataset, 0.25)

# Execute the adaboost algorithm
Adaboost(train, test)
