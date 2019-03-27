using DataFrames
using CSV
using Statistics
using Random
using StatsBase

function gini_impurity(S_ds_imp::DataFrame, Feature::Integer, S_imp::Float64)
    #Number of columns
    m         = size(S_ds_imp,2)
    Y_uniques = unique(S_ds_imp[:,m])

    #Filter just one feature
    data_impurity = S_ds_imp[:,Feature]
    #Distinct values of dataframe
    unique_feature = sort(unique(data_impurity))

    #Means between intervals
    unique_means   = []    
    for i in range(1,length=size(unique_feature,1)-1)
        push!(unique_means, ( unique_feature[i] + unique_feature[i+1] ) / 2 )
    end
    #if size(unique_means, 1) == 0
    #    unique_means = unique_feature
    #end
    
    #Minimun of impurity
    gini_impurity_feature = (0,size(unique_means, 1),0)
    for mean in unique_means

        node_left  = filter(x -> (x[:][Feature] <= mean), S_ds_imp)
        node_right = filter(x -> (x[:][Feature] > mean), S_ds_imp)
        
        gini_impurity_left = 1.0
        if size(node_left,1) > 0  
            for y in range(1, length=size(Y_uniques, 1))
                gini_impurity_left -= ( count(x->(x==Y_uniques[y]),node_left[:,m]) / size(node_left,1) ) ^ 2
            end
        end
                
        gini_impurity_right = 1.0
        if size(node_right,1) > 0 
            for y in range(1, length=size(Y_uniques, 1))
                gini_impurity_right -= ( count(x->(x==Y_uniques[y]),node_right[:,m]) / size(node_right,1) ) ^ 2
            end
        end

        gini_impurity_node = ( ( size(node_left,1) / size(S_ds_imp,1) ) * gini_impurity_left ) + ( ( size(node_right,1) / size(S_ds_imp,1) ) * gini_impurity_right )
        if (gini_impurity_node < gini_impurity_feature[2]) && (gini_impurity_feature[2] > 0.0)
            gini_impurity_feature = (mean, gini_impurity_node, Feature)
        end
    end
    return gini_impurity_feature
end

function gini_impurity_string(S_ds_imp::DataFrame, Feature_X::Integer, S_imp::Float64)
    #Number of columns
    m         = size(S_ds_imp,2)
    Y_uniques = unique(S_ds_imp[:,m])

    #Filter just one feature
    data_impurity = S_ds_imp[:,Feature_X]
    #Distinct values of dataframe
    unique_feature = sort(unique(data_impurity))

    #Minimun of impurity
    gini_impurity_feature = (0,size(unique_feature, 1),0)
    for feature in unique_feature

        node_side = filter(x -> (x[:][Feature_X] == feature), S_ds_imp)
        
        gini_impurity_side = 1.0
        if size(node_side,1) > 0  
            for y in range(1, length=size(Y_uniques, 1))
                gini_impurity_side -= ( count(x->(x==Y_uniques[y]),node_side[:,m]) / size(node_side,1) ) ^ 2
            end
        end

        gini_impurity_node = ( ( size(node_side,1) / size(S_ds_imp,1) ) * gini_impurity_side )
        if (gini_impurity_node < gini_impurity_feature[2]) && (gini_impurity_feature[2] > 0.0)
            gini_impurity_feature = (feature, gini_impurity_node, Feature_X)
        end
    end
    return gini_impurity_feature
end