using Pkg
Pkg.activate(".")
#%%
using MLDatasets, Statistics, LinearAlgebra
#%% Load the MNIS Dataset
# The datasets split attribute will have what is necessary
training_data = MNIST(; split=:train)
test_data = MNIST(; split=:test)
#%% Pull out the features and targets
train_features = training_data.features
train_target = training_data.targets
test_features = test_data.features
test_target = test_data.targets
#%% Flatten the data (already in Float32)
function flatten(features::Array)
    # Assuming 3 dimensional array
    dims = size(features)
    xdims = dims[1]
    ydims = dims[2]
    observation_size = xdims * ydims
    return reshape(features, observation_size, :)
end
X_train = flatten(train_features)
X_test = flatten(test_features)
#%% One-Hot encode the targets
function one_hot(labels::AbstractVector{<:Integer}, num_classes=10)
    # Create an array: (num_classes, number_of_labels)
    y = zeros(Float32, num_classes, length(labels))
    for (i, lbl) in enumerate(labels)
        # lbl is the value of the ith term of labels
        y[lbl + 1, i] = 1.0f0
    end
    return y
end
#%%
y_train = one_hot(train_target)
y_test = one_hot(test_target)
#%% Define an activation function
reLU(x) = max.(x, 0)
function shifted_softmax(logits)
    shifted_array = logits .- maximum(logits; dims=1)
    exps = exp.(shifted_array)
    return exps ./ sum(exps; dims=1)
end
#%% Define a cross-entropy function
function cross_entropy(predicted, target)
    eps = 1e-10f0 # Define a small number to prevent taking the log of 0
    return -mean(sum(target .* log.(predicted .+ eps); dims=1))
end
