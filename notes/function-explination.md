# Neural Networks

## Encoding

### One-Hot Encoding

**Definition:** A way to represent categorical variables as numerical vectors where one element is *hot* (1) and all others are not (0)
```julia
function one_hot(labels::AbstractVector{<:Integer}, num_classes=10)
    # Create an array: (num_classes, number_of_labels)
    y = zeros(Float32, num_classes, length(labels))
    for (i, lbl) in enumerate(labels)
        # lbl is the value of the ith term of labels
        y[lbl+1, i] = 1.0f0
    end
    return y
end
```
Taking in an abstract vector (in this case of integers because I am clarifying numbers)

### Cross-Entropy Loss
Cross-Entropy measures how well a predicted probability distribution matches the true distributions
#### Example
If [[2-Notes/1-Concepts/neural-networks|Neural Networks#Softmax]]  is the predicted distribution, and [[neural-networks|Neural Networks#One-Hot Encoding]] is the actual distribution, you have something like this:

A **Predicted Distribution** $\hat{y} = \langle p_1, p_2, \dots, p_K \rangle | \forall p_i \in [0,1], \sum_{i=1}^K p_i \equiv 1$
Where, the actual distribution is structured as $y = \langle 0, \dots, 1, \dots, 0\rangle$ where 1 is in the correct position

Cross-Entropy measures the Distance between these two distributions:
$$
H(y, \hat{y}) = - \sum_{i=1}^K y_i \log(p_i)
$$

With a [[neural-networks#One-Hot Encoding]], $y_i$ will be nonzero only in the correct place, so $H$ will be $-\log(p_i)$
As $\log(1) =0$, $H$ will approach 0 as the model improves

#### Need to Avoid taking the Log of Zero

When implementing cross-entropy, it is important to not take the log of 0 (which is undefined approaching negative infinity), so you add in an `eps` variable to prevent it

```julia
function cross_entropy(predicted, target)
    eps = 1e-10f0 # Define a small number to prevent taking the log of 0
    return -mean(sum(target .* log.(predicted .+ eps); dims=1))
end
```

## Activation

### ReLU
**Definition:** Rectified Linear Unit activation
An incredibly simple activation function:
$$
\text{ReLU}(x) = \begin{cases}
    0 & x \leq 0 \\ 
    x & x > 0 
    \end{cases}
$$
```julia
ReLU(x) = max.(x, 0)
```
## Methods

### Softmax

**Definition:** Given a vector of real numbers, the softmax function transforms these values into non-negative numbers that sum to 1, interpreting them as probabilities
$$
f(z_i) = \frac{\exp\{z_i\}}{\sum_{j=1}^K \exp\{z_i\}}
$$
Where $z_i$ is an element of a vector with length $K$

#### Shifted

Because exponentials shift uniformly, and because Float32s (the most commonly used type in ML), it is common to subtract the maximum element in an array from each element.
```julia
function shifted_softmax(logits)
    shifted_array = logits .- maximum(logits, dims=1)
    exps = exp.(shifted_array)
    return exps ./ sum(exps, dims=1)
end
```
