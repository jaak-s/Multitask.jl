module Multitask

export nuclearNormMT
export nuclearNormMTcol
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
        u, v, s = power(D, 10)
        ## 3. add new solution
        α = 2.0 / (i + 2)
        M = (1 - α) * M + α * tau * u * v'
    end
    return M
end

function nuclearNormMTSparse(Xw,
                     Yw::Array{Array{Float64,1},1},
                     tau,
                     lambda=0.0, 
                     nsteps=200)
    Nfeat    = size(Xw[1], 1)
    Ntasks   = length(Xw)
    M = zeros(Nfeat, Ntasks)
    D = zeros(M)
    Xnz = map(x -> findnz(x), Xw)
    for i = 0:(nsteps-1)
        ## 1. negative gradient
        for t = 1:Ntasks
            #D[:,t] = - 2.0Xw[t] * (Xw[t]' * M[:,t] - Yw[t])
            setSparseProd!(D, t, Xnz[t][1], Xnz[t][2], Xnz[t][3], -2.0*(Xw[t]' * M[:,t] - Yw[t]) )
            if lambda > 0
              D[:,t] += - 2.0lambda * M[:,t]
            end
        end
        ## 2. find 1st singular vectors
        u, v, s = power(D, 10)
        ## 3. add new solution
        α = 2.0 / (i + 2)
        M = (1 - α) * M + α * tau * u * v'
    end
    return M
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
