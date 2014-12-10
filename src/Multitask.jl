module Multitask

export nuclearNormMT
export nuclearNormMTSparse
export power

function power(A, n)
    x = randn(size(A,1))

    for i = 1:n
        x = x / maximum(abs(x))
        x = A*(A'*x)
    end
    x = x / norm(x)
    v = A'*x
    s = norm(v)
    return x, v/s, s
end

function nuclearNormMT(Xw,
                     Yw::Array{Array{Float64,1},1},
                     tau,
                     lambda=0.0, 
                     nsteps=200)
    Nfeat    = size(Xw[1], 2)
    Ntasks   = length(Xw)
    M = zeros(Nfeat, Ntasks)
    D = zeros(M)
    for i = 0:(nsteps-1)
        ## 1. negative gradient
        for t = 1:Ntasks
            D[:,t] = - 2.0Xw[t]' * (Xw[t] * M[:,t] - Yw[t])
            if lambda > 0
              D[:,t] += - 2.0lambda * M[:,t]
            end
        end
        ## 2. find 1st singular vectors
        u, v, s = power(D, 20)
        ## 3. add new solution
        α = 2.0 / (i + 2)
        M = (1 - α) * M + α * tau * u * v'
    end
    return M
end

type MTFit
  M::Matrix{Float64}
  alphas::Vector{Float64}
  sqerrors::Vector{Float64}
  tau::Float64
end

function nuclearNormMTSparse(Xw,
                     Yw::Array{Array{Float64,1},1},
                     tau;
                     lambda=0.0, 
                     nsteps=200,
                     bestStep=true,
                     balanceTasks=true)
    Nfeat  = size(Xw[1], 1)
    Ntasks = length(Xw)
    M = zeros(Nfeat, Ntasks)
    D = zeros(M)
    Xnz = map(x -> findnz(x), Xw)
    ## computing task weights
    lambdaN = lambda
    weights = ones(Ntasks)
    if balanceTasks
        Ns      = map(y -> length(y), Yw)
        weights = map(n -> 1.0/n, Ns)
        Naver   = sum(Ns) / Ntasks
        lambdaN = lambda / Naver
    end
    alphas   = Float64[]
    sqerrors = Float64[]
    for i = 0:(nsteps-1)
        ## 1. negative gradient
        for t = 1:Ntasks
            #D[:,t] = - 2.0Xw[t] * (Xw[t]' * M[:,t] - Yw[t])
            setSparseProd!(D, t, Xnz[t][1], Xnz[t][2], Xnz[t][3], -2.0*weights[t]*(Xw[t]' * M[:,t] - Yw[t]) )
            if lambda > 0
              D[:,t] += - 2.0lambdaN * M[:,t]
            end
        end
        ## 2. find 1st singular vectors
        u, v, s = power(D, 20)
        ## 3. add new solution
        if bestStep
            n = 0.0
            d = 0.0
            sq = 0.0
            for t = 1:Ntasks
                yhat = Xw[t]' * M[:,t]
                Xq   = Xw[t]' * (v[t] * tau * u)
                n  += weights[t] * (yhat - Yw[t])' * (yhat - Xq)
                d  += weights[t] * sum( (yhat - Xq).^2 )
                sq += weights[t] * sum( (yhat - Yw[t]) .^ 2 )
                if lambda > 0
                  m_q = M[:,t] - v[t] * tau * u
                  n  += lambdaN * M[:,t]' * m_q
                  d  += lambdaN * sum(m_q .^ 2)
                  sq += lambdaN * sum(M[:,t] .^ 2)
                end
            end
            α = min(1, max(0, n[1] / d[1])) 
            push!(sqerrors, sq)
        else
            α = 2.0 / (i + 2)
        end
        push!(alphas, α)
        M *= (1 - α)
        for t = 1:Ntasks
          M[:,t] += α * tau * v[t] * u
        end
    end
    return MTFit(M, alphas, sqerrors, tau)
end

## computes res[:,col] = X*w
function setSparseProd!(res::Array{Float64,2}, col, xi, xj, xv, w)
    for i = 1:size(res, 1)
        res[i, col] = 0.0
    end
    for i = 1:size(xi, 1)
        res[ xi[i], col ] += xv[i] * w[xj[i]]
    end
end

end # module
