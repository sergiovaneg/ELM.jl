import LinearAlgebra
import Random

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

    function RBF_neuron{T}(d::Integer, Mu::Vector{T} = [], Sigma::Vector{T} = [], act_fcn::Function = sigmoid) where T<:Real
        a = Vector{T}(undef, d);
        if isempty(Mu) || isempty(Sigma)
            Random.rand!(a);
            a .*= 2.;
            a .-= 1;
        else
            Random.randn!(a);
            a *= Sigma;
            a += Mu;
        end

        b = 2 * Random.rand(T) - 1;

        new(a, b, act_fcn);
    end 
end

function (neuron::RBF_neuron)(x::Vector{T}) where T<:Real
    return neuron.act_fcn(neuron.b * LinearAlgebra.norm(x - neuron.a));
end

struct ADD_neuron{T<:Real} <: AbstractNeuron{T}
    a::Vector{T}
    b::T
    act_fcn::Function

    function ADD_neuron{T}(d::Integer, Mu::Vector{T} = [], Sigma::Vector{T} = [], act_fcn::Function = sigmoid) where T<:Real
        a = Vector{T}(undef, d);
        if isempty(Mu) || isempty(Sigma)
            Random.rand!(a);
            a .*= 2.;
            a .-= 1;
        else
            Random.randn!(a);
            a *= Sigma;
            a += Mu;
        end

        b = 2 * Random.rand(T) - 1;
        
        new(a, b, act_fcn);
    end
end

function (neuron::ADD_neuron)(x::Vector{T}) where T<:Real
    return neuron.act_fcn(LinearAlgebra.dot(x, a) + b);
end