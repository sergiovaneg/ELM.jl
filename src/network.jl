include("neurons.jl")

mutable struct Network{T<:Real}
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

    function Network{T}(L::Int, input_vars::Tuple{String, Vararg{String}},
        output_vars::Tuple{String, Vararg{String}}; input_delays::Tuple{Integer, Vararg{Integer}} = (0,),
        fb_delays::Tuple{Vararg{Integer}} = Tuple{}(), act_fcn::Function = (x::T -> x), 
        output_transform::Function = (x::T -> x), 
        inv_output_transform::Function = (x::T -> x)) where T<:Real

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

Base.convert(::Type{Bool}, net::Network) = net.L;

function apply_delay(self::Network{T}, X::Matrix{T}) where T<:Real
    if !self.offset
        return copy(X);
    end

    X_ = Matrix{T}(undef, size(X, 1), 0);

    for var_idx = 1:length(self.input_vars)
        aux_data = [fill(X_[1, var_idx], (self.offset, 1)); X_[:, var_idx]];
        for delay ∈ (self.input_vars[var_idx] ∈ self.output_vars ? self.fb_delays : self.input_delays)
            X_ = [X_ aux_data[(1+self.offset)-delay:end-delay]];
        end
    end

    return X_;
end

function normalize_input(self::Network{T}, X::Matrix{T}) where T<:Real
    X_ = copy(X);
    for output_var ∈ self.output_vars
        if !(output_var ∈ self.input_vars)
            continue;
        end
        
        X_[:, findfirst(isequal(output_var), self.input_vars)] = self.output_transform.(X_[:, findfirst(isequal(output_var), self.input_vars)]);
    end

    for var_idx = 1:length(self.input_vars)
        X_[:, var_idx] = (X_[:, var_idx] .- self.input_bias[var_idx]) ./ self.input_scale[var_idx];
    end

    return X_;
end

function normalize_output(self::Network{T}, Y::Matrix{T}) where T<:Real
    Y_ = similar(Y);

    for var_idx = 1:length(self.output_vars)
        Y_[:, var_idx] = (self.output_transform.(Y[:, var_idx]) .- self.output_bias[var_idx]) ./ self.output_scale[var_idx];
    end

    return Y_;
end

function denormalize_output(self::Network{T}, Y_::Matrix{T}) where T<:Real
    Y = similar(Y_);

    for var_idx = 1:length(self.output_vars)
        Y[:, var_idx] = self.inv_output_transform.(Y_[:, var_idx] .* self.output_scale[var_idx] .+ self.output_bias[var_idx]);
    end

    return Y;
end