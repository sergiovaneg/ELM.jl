import LinearAlgebra

function sigmoid(x::Real)
    return 1.0 / (1.0 + exp(-x))
end

function gaussian(x::Real)
    return exp(-(x ^ 2))
end

abstract type AbstractNeuron{T<:Real} end

struct RBF_neuron{T<:Real} <: AbstractNeuron{T}
    a::Vector{T}
    b::T
    act_fcn::Function
    g::Function

    function RBF_neuron{T}(a::Vector{T}, b::T, act_fcn::Function) where T<:Real
        g(x::Vector{T}) = act_fcn(b * LinearAlgebra.norm(x - a));
        new(a, b, act_fcn, g);
    end 
end

struct ADD_neuron{T<:Real} <: AbstractNeuron{T}
    a::Vector{T}
    b::T
    act_fcn::Function
    g::Function

    function ADD_neuron{T}(a::Vector{T}, b::T, act_fcn::Function) where T<:Real
        g(x::Vector{T}) = act_fcn(LinearAlgebra.dot(x, a) + b);
        new(a, b, act_fcn, g);
    end
end