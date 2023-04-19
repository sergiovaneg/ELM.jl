module ELM

export ELM_regressor, fit!
include("networks.jl")

export sigmoid, gaussian, RBF_neuron, ADD_neuron
include("neurons.jl")

end