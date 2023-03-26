import LinearAlgebra

function sigmoid(x::Array{Float64,1})
    return 1.0 ./ (1.0 .+ exp.(-x))
end

function gaussian(x::Array{Float64,1})
    return exp.(-(x .^ 2))
end

abstract type AbstractNeuron{T <: Real} end

struct RBF_neuron{T <: Real} <: AbstractNeuron{T}
    a::Vector{T}
    b::T
    act_fcn::Function
    g::Function

    function RBF_neuron{T}(a::Vector{T}, b::T, act_fcn::Function) where T<:Real
        g(x::Matrix{T}) = act_fcn.(b.*LinearAlgebra.norm.(eachrow(x.-a')));
        new(a, b, act_fcn, g);
    end 
end