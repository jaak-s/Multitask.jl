module Multitask

export multitaskFW
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

function multitaskFW(Xw::Array{Array{Float64,2},1},
                     Yw::Array{Array{Float64,1},1},
                     tau,
                     lambda=0.0, 
                     nsteps=100)
    Nfeat    = size(Xw[1], 2)
    Ntasks   = length(Xw)
    M = zeros(Nfeat, Ntasks)
    D = zeros(M)
    for i = 0:(nsteps-1)
        ## 1. negative gradient
        for t = 1:Ntasks
            D[:,t] = - 2.0Xw[t]' * (Xw[t] * M[:,t] - Yw[t])
            if lambda > 0
              D[:,t] -= - 2.0lambda * M[:,t]
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

end # module
