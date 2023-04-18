include("neurons.jl")

import Statistics

mutable struct ELM_regressor{T<:Real}
    d::Integer
    L::Integer
    m::Integer

    input_vars::Tuple{String, Vararg{String}}
    output_vars::Tuple{String, Vararg{String}}
    fb_map::Vector{Tuple{Integer, Integer}}

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

    function ELM_regressor{T}(L::Int, input_vars::Tuple{String, Vararg{String}},
        output_vars::Tuple{String, Vararg{String}}; input_delays::Tuple{Integer, Vararg{Integer}} = (0,),
        fb_delays::Tuple{Vararg{Integer}} = Tuple{}(), act_fcn::Function = (x::T -> x), 
        output_transform::Function = (x::T -> x), 
        inv_output_transform::Function = (x::T -> x)) where T<:Real

        fb_map = [];
        for var_idx ∈ eachindex(output_vars)
            if !(output_vars[var_idx] ∈ self.input_vars)
                continue;
            end
            
            append!(fb_map, (var_idx, findfirst(isequal(output_vars[var_idx]), self.input_vars)));
        end

        offset = Set(input_vars) ∩ Set(output_vars) == ∅ ? max(input_delays) : max(vcat(input_delays, fb_delays));

        G = Vector{AbstractNeuron{T}}();
        beta = Vector{T}();
        P = Matrix{T}();

        input_bias = Vector{T}();
        input_scale = Vector{T}();
        output_bias = Vector{T}();
        output_scale = Vector{T}();

        new(-1, L, -1, input_vars, output_vars, fb_map, input_delays, fb_delays, offset, G, act_fcn, beta,
            P, output_transform, inv_output_transform, input_bias, input_scale, output_bias, output_scale);
    end
end

Base.convert(::Type{Bool}, net::ELM_regressor) = net.L;

function apply_delay(self::ELM_regressor{T}, X::Matrix{T}) where T<:Real
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

function normalize_input(self::ELM_regressor{T}, X::Matrix{T}) where T<:Real
    X_ = copy(X);
    for pair ∈ self.fb_map
        X_[:, pair[1]] = self.output_transform.(X_[:, pair[1]]);
    end

    for var_idx = 1:length(self.input_vars)
        X_[:, var_idx] = (X_[:, var_idx] .- self.input_bias[var_idx]) ./ self.input_scale[var_idx];
    end

    return X_;
end

function normalize_output(self::ELM_regressor{T}, Y::Matrix{T}) where T<:Real
    Y_ = similar(Y);

    for var_idx = 1:length(self.output_vars)
        Y_[:, var_idx] = (self.output_transform.(Y[:, var_idx]) .- self.output_bias[var_idx]) ./ self.output_scale[var_idx];
    end

    return Y_;
end

function denormalize_output(self::ELM_regressor{T}, Y_::Matrix{T}) where T<:Real
    Y = similar(Y_);

    for var_idx = 1:length(self.output_vars)
        Y[:, var_idx] = self.inv_output_transform.(Y_[:, var_idx] .* self.output_scale[var_idx] .+ self.output_bias[var_idx]);
    end

    return Y;
end

function HL_init!(self::ELM_regressor, X::NTuple{N,Matrix{T}}, Y::NTuple{N,Matrix{T}}, neuron_type::Type{<:AbstractNeuron{T}} = RBF_neuron) where {N, T<:Real}
    X_ = vcat(X...);
    for pair ∈ self.fb_map
        X_[:, pair[1]] = self.output_transform.(X_[:, pair[1]]);
    end
    Y_ = self.output_transform.(vcat(Y...));

    # (-1,1) Normalized
    self.input_bias = 0.5*(maximum(X_; dims=1) + minimum(X_; dims=1));
    self.input_scale = 0.5*(maximum(X_; dims=1) - minimum(X_; dims=1));
    self.output_bias = 0.5*(maximum(Y_; dims=1) + minimum(Y_; dims=1));
    self.output_scale = 0.5*(maximum(Y_; dims=1) - minimum(Y_; dims=1));

    X_ = (X_ .- self.input_bias) ./ self.input_scale;
    Y_ = (Y_ .- self.output_bias) ./ self.output_scale;

    Mu = Vector{T}();
    Sigma = Vector{T}();
    for (input_var, mu, sigma) in zip(self.input_vars, Statistics.mean(X_, dims=1), Statistics.std(X_, dims=0))
        if input_var ∈ self.output_vars
            append!(Mu, mu*ones(T, length(self.fb_delays)));
            append!(Sigma, sigma*ones(T, length(self.fb_delays)));
        else
            append!(Mu, mu*ones(T, length(self.input_delays)));
            append!(Sigma, sigma*ones(T, length(self.input_delays)));
        end
    end

    self.d = length(Mu);
    self.G = [neuron_type(self.d, Mu, Sigma, self.act_fcn) for _ in range(self.L)];

    self.m = size(Y_, 2);
    self.beta = zeros(self.L+1, self.m);
end

function fit!(self::ELM_regressor, X::NTuple{N,Matrix{T}}, Y::NTuple{N,Matrix{T}}, Lambda::T = 1e18, Epsilon::T = 1e-9) where {N, T<:Real}
    if isempty(self.G)
        HL_init!(self, X, Y);
    end

    X_norm_delay = [self.apply_delay(self.normalize_input(X_single)) for X_single in X];
    Y_norm = [self.normalize_output(Y_single) for Y_single in Y];

    if isempty(self.P)
        H_0 = LinearAlgebra.UniformScaling(Epsilon);
        Y_0 = zeros(T, self.L + 1, self.m);

        self.P = inv(LinearAlgebra.UniformScaling(1/Lambda) + H_0' * H_0);
        self.beta = (LinearAlgebra.UniformScaling(1/Lambda) + H_0' * H_0) \ (H_0' * Y_0);
    end

    H = [hcat([G(X_norm_delay_single) for G in self.G]..., ones(T, size(X, 1), 1)) for X_norm_delay_single in X_norm_delay];
    for (H_single, Y_norm_single) in zip(H, Y_norm)
        self.P -= self.P * H_single' * ((LinearAlgebra.I + H_single * self.P * H_single') \ (H_single * self.P));
        self.beta += self.P * H_single' * (Y_norm_single - H_single * self.beta);
    end

    return self;
end