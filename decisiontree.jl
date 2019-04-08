using DataFrames
using CSV
using Statistics
using Random
using StatsBase

include("lib/gini_impurity.jl")
include("lib/helpers.jl")

using DataStructures

# Show how data - nodes are structed
mutable struct Node
    data::DataFrame
    index::Int64
    gini::Float64
    isLeaf::Bool
    way::String
    mean::Any
    feature::Int64
    nodeTrue::Node
    nodeFalse::Node
    Node(data,index,gini,isLeaf,way,mean,feature) = new(data,index,gini,isLeaf,way,mean,feature)
end

# Show how data - trees are structed
mutable struct Tree
    nodes::Array{Node}
    result::String
    n_cols::Int64
    order_cols::Array{Int64}
    Tree() = new()
end

function getMin(ItemDict, Index)
    min     = Inf
    ret_min = (Integer,Any,Float64)
    for ItemArray in ItemDict
        for Item in ItemArray[2]
            if Item[Index] < min
                min = Item[Index]
                ret_min = Item
            end
        end        
    end
    return ret_min
end

function getMaxOccur(items)
    max   = 0
    occur = ""
    for (key, value) in items
        if value > max
            occur = key
            max = value
        end
    end
    return occur
end

function BuildTree(S::DataFrame, NodeFrom::Node, Nodes::Array{Node}, Position::Integer = 0, GiniImpurity::Float64 = 1.0, Way::String = "Root")
    # Get the node
    features_impurity = Dict(
        "String" => Array{Tuple{Integer,String,Float64}}(undef,0),
        "Float64" => Array{Tuple{Integer,Float64,Float64}}(undef,0)
    )
    for j in range(1,length=size(S,2)-1)        
        if (eltypes(S)[j].b == String)
            mean_imp, gini_imp = gini_impurity_string(S, j, GiniImpurity)
            push!(features_impurity["String"], (j, mean_imp, gini_imp))
        else
            mean_imp, gini_imp = gini_impurity(S, j, GiniImpurity)
            push!(features_impurity["Float64"], (j, mean_imp, gini_imp))
        end
    end
    node_min = getMin(features_impurity,3)    
    
    if (size( unique(S[:,size(S,2)]) , 1 ) > 1) && (size( unique( S[:,node_min[1]] ), 1) > 1)
        node = Node(S, Position, node_min[3], false, Way, node_min[2], node_min[1])
        # Go to left - true
        BuildTree(filterDS(S, true, node_min), node, Nodes, Position + 1, 1., "True")
        # Go to right - false
        BuildTree(filterDS(S, false, node_min), node, Nodes, Position + 1, 1., "False")
    else
        node = Node(S, Position, node_min[3], true, Way, node_min[2], node_min[1])
    end
    if Way == "True"
        NodeFrom.nodeTrue = node
    end
    if Way == "False"
        NodeFrom.nodeFalse = node
    end

    push!(Nodes, node)    
end

function prediction(Test, nodes)
    node = getRoot(nodes)
    while !node.isLeaf
        if check_predict(Test[node.feature], node.mean)
            # True
            node = node.nodeTrue
        else
            # False
            node = node.nodeFalse
        end
    end
    return unique(node.data[:,size(Test,1)])[1]
end

function getRoot(Nodes)
    for node in Nodes
        if node.way == "Root"
            return node
        end
    end
end

function decision_tree(S::DataFrame)
    
    # Start the root node of tree and array of Nodes
    node_root   = Node(S, 0, 0.,false, "None", 0., 0)
    nodes       = Array{Node}(undef,0)
    
    # Start building and trainig the tree
    BuildTree(S, node_root, nodes, 0, 1.0)
    
    return nodes
end