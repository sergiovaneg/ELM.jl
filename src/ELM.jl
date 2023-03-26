module ELM

import Statistics
import DSP
import Clustering
import Random
import LinearAlgebra

mutable struct network{T <: Real}
    G::Vector{TypeNeuron{T}}
    act_fcn::Function

    beta::Vector{T}
    P::Array{T, 2}

    d::Int
    L::Int
    m::Int

    input_bias::Vector{T}
    input_scale::Vector{T}
    output_bias::Vector{T}
    output_scale::Vector{T}

    Mu::Vector{T}
    Sigma::Vector{T}

    input_vars::Tuple{str, Vararg{string}}
    output_vars::Tuple{str, Vararg{string}}

    input_delays::Tuple{Integer, Vararg{Integer}}
    fb_delays::Tuple{Integer, Vararg{Integer}}
    offset::Int

    output_transform::Function
    inv_output_transform::Function

    critical_value::Float64

    function network{T}(L::Int, ) where T <: Real
end

end
