# Multitask

[![Build Status](https://travis-ci.org/jaak-s/Multitask.jl.svg?branch=master)](https://travis-ci.org/jaak-s/Multitask.jl)

Multi-task learning using nuclear norm constraint for least squares regression.
The solver is based on Conditional Gradient Descent (also known as generalized
Frank-Wolfe method).

## Installation
```julia
Pkg.clone("https://github.com/jaak-s/Multitask.jl.git")
```

## Example usage
```julia
## generating random model and data from it
using Distributions
Nfeat = 1000
Ntasks = 100
Nsamples = 500
Nknown = 50:100
Ncore = 10
Mcore = randn(Nfeat, Ncore)
U = randn(Ncore, Ntasks)
## each column in M is a linear combination of Mcore vectos
M = Mcore * U
## setup samples
W = Vector{Int64}[]
for p = 1:Ntasks
    push!(W, sample(1:Nsamples, rand(Nknown), replace=false))
end
## generate samples
X = randn(Nsamples, Nfeat)
Y = X * M
## using only known samples for training
Xtrain = map(w -> X[w,:], W);
Ytrain = map(t -> Y[W[t], t], 1:length(W));
```

Now we can build nuclear norm constrained model.
```julia
## running nuclear norm MTL method
using Multitask
tau    = 2400
lambda = 0
Nsteps = 200
Mhat = nuclearNormMT(Xtrain, Ytrain, tau, lambda, Nsteps)

## predicting whole Y matrix
Yhat = X * Mhat

## accuracy on whole Y matrix:
sse  = sum( (Y - Yhat).^2 )
Yvar = sum( (Y - mean(Y)).^2 )
println("Sum SE: ", sse ) 
println("Var(Y): ", Yvar )
println("R^2   : ", 1 - sse/Yvar)
```

For comparison we train ridge regression for each task, i.e. single task learning.
```julia
## regression
function regression(X, y, lambda)
    (X' * X + lambda * eye(size(X,2))) \ (X' * y)
end
M1hat = zeros(M)
for t = 1:length(Xtrain)
    M1hat[:,t] = regression(Xtrain[t], Ytrain[t], 1);
end
## predict the whole matrix
Y1hat = X * M1hat;

## accuracy on whole Y matrix:
sse_reg  = sum( (Y - Y1hat).^2 )
Yvar     = sum( (Y - mean(Y)).^2 )
println("Regr SSE: ", sse_reg ) 
println("Regr R^2: ", 1 - sse_reg/Yvar)
```

The values of R^2 for multi-task learning are around 0.66 and for single task 0.22.
