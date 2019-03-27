using DataFrames
using CSV
using Statistics
using Random
using StatsBase

function filterDS(S::DataFrame, Equal::Bool, node_min::Tuple{Integer,Any,Float64})
    
    if (typeof(node_min[2]) == String)
        if (Equal)
            return filter(x -> x[:][node_min[1]] == node_min[2],S)
        else
            return filter(x -> x[:][node_min[1]] != node_min[2],S)
        end
    end
    if (typeof(node_min[2]) == Float64)
        if (Equal)
            return filter(x -> x[:][node_min[1]] <= node_min[2],S)
        else
            return filter(x -> x[:][node_min[1]] > node_min[2],S)
        end
    end
    if (typeof(node_min[2]) == Int64)
        if (Equal)
            return filter(x -> x[:][node_min[1]] <= node_min[2],S)
        else
            return filter(x -> x[:][node_min[1]] > node_min[2],S)
        end
    end

end

function check_predict(row_feature, mean)
    if (typeof(mean) == String)
        return (row_feature == mean)
    end
    if (typeof(mean) == Float64)
        return (row_feature <= mean)
    end
    if (typeof(mean) == Int64)
        return (row_feature <= mean)
    end
end

function AccuracyDT(predictions::Array{Bool})
    perc_right_answer = size(filter(x -> x == true, predictions), 1) / size(predictions,1)
    return perc_right_answer * 100
end