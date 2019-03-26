using DataFrames

function Main(x_train, x_test, y_train, y_test, max_iterations = 5)
    # Initialize W - weights = 1/max_iterations
    W = [1/size(x_train,1) for iteration in range(1,length=size(x_train,1))]
    for iteration in range(1, length=max_iterations)
        # Local distribuiton from train
        St = x_train
        # Train the Dt distribuiton
        h = train(St)
        # Compute the error Et 
        Et += ( ( ( y_train[i] == h(i) ) ? 1 : 0 for i in range(1,length=size(St,1))) / W[iteration])
        # Compute the new weight
        Wt = Compute_Weight(Et)
    end
end

function Compute_Weight(error_weight)
    return 0.5 * log((1-error_weight)/error_weight)
end
