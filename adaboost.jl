using DataFrames
using CSV

include("decisiontree.jl")

function AdaboostOld(train, test, max_iterations = 5)
    # train x, y
    x_train = train[:,1:size(train,2)-1]
    y_train = train[:,size(train,2):size(train,2)]
    
    # test x, y
    x_test = test[:,1:size(test,2)-1]
    y_test = test[:,size(test,2):size(test,2)]

    # size of DataSet
    N = size(x_train, 1)
    
    # initialize W => equal of number of observations
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

function Adaboost(train, test, max_iterations = 5)
    # size of DataSet
    N = size(train, 1)
    
    # initialize W => equal of number of observations
    W = [ [1 / N for i in range(1, length=N)] for i in range(1, length=max_iterations)]
    for iteration in range(1, length=max_iterations - 1)
        Compute_Distribuiton(train, W, iteration)
    end

    @show W
end

function Compute_Distribuiton(train, W, iteration)
    
    # train the x_train_i dataset
    nodes = train_distribuiton(train)
    
    # compute e_i
    e_i =  sum( W[iteration][i] * is_correct( h(train[i,:],nodes), train[i,size(train, 2)] ) for i in range(1, length=size(train, 1)) )
    e_i /= sum( W[iteration][i] for i in range(1, length=size(train, 1)) )
    
    # compute votes
    omega_i = 0.5 * log( (1 - e_i) / e_i )
    @show omega_i
    # update weight
    W[iteration + 1] = [ W[iteration][i] * exp( (-1 * omega_i) * train[i,size(train,2)] * h(train[i,:],nodes) ) for i in range(1, length=size(train,1)) ]
    @show W
    # normalize W
    Z_i = maximum(W[iteration])
    W[iteration + 1] = [ W[iteration + 1][i] / Z_i for i in range(1,length=size(train,1))]
    @show W
end

function is_correct(a, b)
    return (a==b) ? 0 : 1
end

function train_distribuiton(train)
    return decision_tree(train)
end

function h(x, nodes)
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
