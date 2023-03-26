include("neurons.jl")

mutable struct network{T <: Real}
    d::Integer
    L::Integer
    m::Integer

    input_vars::Tuple{String, Vararg{String}}
    output_vars::Tuple{String, Vararg{String}}

    input_delays::Tuple{Integer, Vararg{Integer}}
    fb_delays::Tuple{Vararg{Integer}}
    offset::Int

    G::Vector{AbstractNeuron{T}}
    act_fcn::Function

    beta::Vector{T}
    P::Matrix{T}

    output_transform::Function
    inv_output_transform::Function

    input_bias::Vector{T}
    input_scale::Vector{T}
    output_bias::Vector{T}
    output_scale::Vector{T}

    critical_value::Real

    function network{T}(L::Int, input_vars::Tuple{String, Vararg{String}},
        output_vars::Tuple{String, Vararg{String}}; input_delays::Tuple{Integer, Vararg{Integer}} = (0,),
        fb_delays::Tuple{Vararg{Integer}} = Tuple{}(), act_fcn::Function = (x::T -> x), 
        output_transform::Function = (x::T -> x), 
        inv_output_transform::Function = (x::T -> x)) where T <: Real

        offset = Set(input_vars) ∩ Set(output_vars) == ∅ ? max(input_delays) : max(vcat(input_delays, fb_delays));

        G = Vector{AbstractNeuron{T}}();
        beta = Vector{T}();
        P = Matrix{T}();

        input_bias = Vector{T}();
        input_scale = Vector{T}();
        output_bias = Vector{T}();
        output_scale = Vector{T}();

        new(-1, L, -1, input_vars, output_vars, input_delays, fb_delays, offset, G, act_fcn, beta,
            P, output_transform, inv_output_transform, input_bias, input_scale, output_bias, output_scale, 1.);
    end
end

Base.convert(::Type{Bool}, net::network) = net.L;