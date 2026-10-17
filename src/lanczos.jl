
_resolve_eps(eps_factor, T) = isnothing(eps_factor) ? 1e-4*sqrt(eps(T)) : T(eps_factor) * sqrt(eps(T))


"""
    build_liouvillian_eigenbasis(H, oper; tol=1e-10)

Diagonalize H and project `oper` into the eigenbasis to obtain a Constraint
representation of the Liouvillian suitable for vectorial Lanczos.

Steps:
1. Diagonalize H → eigenvalues E, eigenvectors U
2. Transform oper: Õ = U' * oper * U
3. Form all transition frequencies ωₘₙ = Eₘ - Eₙ and weights |Õₘₙ|
4. Sort by frequency, collapse degenerate blocks (norm of weights)
5. Drop zero-weight frequencies

# Returns
- `E_eff::Vector{Float64}`: effective transition frequencies
- `V_eff::Vector{Float64}`: effective operator weights
"""
# ----- C routine for faster code ------

# Compute dot(a, b) into pre-allocated acc, using tmp as scratch -- zero allocations
function _bf_dot!(acc::BigFloat, tmp::BigFloat, a::Vector{BigFloat}, b::Vector{BigFloat})
    rnd = Base.MPFR.ROUNDING_MODE[]
    ccall((:mpfr_set_ui, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Culong, Base.MPFR.MPFRRoundingMode), acc, 0, rnd)
    @inbounds for i in eachindex(a)
        ccall((:mpfr_mul, Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              tmp, a[i], b[i], rnd)
        ccall((:mpfr_add, Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              acc, acc, tmp, rnd)
    end
end

# y[i] -= a * x[i] for all i, using tmp as scratch -- zero allocations
function _bf_axpy_neg!(a::BigFloat, x::Vector{BigFloat}, y::Vector{BigFloat}, tmp::BigFloat)
    rnd = Base.MPFR.ROUNDING_MODE[]
    @inbounds for i in eachindex(x)
        ccall((:mpfr_mul, Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              tmp, a, x[i], rnd)
        ccall((:mpfr_sub, Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              y[i], y[i], tmp, rnd)
    end
end

# ---- Core of Code -----
function build_liouvillian_eigenbasis(H::AbstractMatrix{Complex{T}}, 
    oper::AbstractMatrix{Complex{T}};
    tol_factor               =nothing,
    remove_degeneracies::Bool=true) where {T<:AbstractFloat}
     
    tol = _resolve_eps(tol_factor, T)
    _run() = begin
        H_d    = Hermitian(Matrix(H))
        O_d = Matrix(oper)
        E, U = eigen(H_d)

        O_tilde = U' * O_d * U

        N       = length(E)
        freqs   = vec([E[m] - E[n] for m in 1:N, n in 1:N])
        weights = vec(abs.(O_tilde))

        perm    = sortperm(freqs)
        freqs   = freqs[perm]
        weights = weights[perm]

        # Find degenerate frequency blocks and collapse their weights
        starts = [1; findall(i -> !isapprox(freqs[i], freqs[i+1]; rtol=tol, atol=tol),
                             1:length(freqs)-1) .+ 1]
        ends   = [starts[2:end] .- 1; length(freqs)]

        collapsed = [norm(weights[s:e]) for (s, e) in zip(starts, ends)]


        if remove_degeneracies
            keep   = collapsed .> tol
            E_eff  = freqs[starts[keep]]
            V_eff  = collapsed[keep]
        else
            E_eff  = freqs                # one representative per block (or all)
            V_eff  = weights           # keep even zero-weight blocks
        end

        @info "Liouvillian eigenbasis built" n_freqs=length(E_eff) original_dim=N^2 precision=@sprintf("%.2e", Float64(tol)) remove_degeneracies=remove_degeneracies
        return E_eff, V_eff
    end
    if T === BigFloat
        # Read precision from the stored values, not the global default
        bf_bits = precision(real(first(nonzeros(H))))
        return setprecision(_run, bf_bits)
    else
        return _run()
    end
end


"""
    _reortho_fro!(vs, n, A)

Full single-pass (Gram-Schmidt) reorthogonalization of `A` against
`vs[1..n-1]`.  Uses BLAS `axpy!` for Float64/Double64.
"""
function _reortho_fro!(vs::Vector, n::Int, A::Vector)
    for k in 1:n-1
        axpy!(-dot(vs[k], A), vs[k], A)
    end
end

"""
    _reortho_dro!(vs, n, A)

Double-pass reorthogonalization: two sequential Gram-Schmidt sweeps.
More stable than a single pass for ill-conditioned bases.
"""
function _reortho_dro!(vs::Vector, n::Int, A::Vector)
    for _ in 1:2
        for k in 1:n-1
            axpy!(-dot(vs[k], A), vs[k], A)
        end
    end
end

"""
    _reortho_fro_bf!(vs, n, A, dot_acc, dot_tmp)

BigFloat single-pass reorthogonalization using direct MPFR C calls to avoid
allocations.  `dot_acc` and `dot_tmp` are pre-allocated scratch BigFloats.
"""
function _reortho_fro_bf!(vs::Vector, n::Int, A::Vector{BigFloat},
                            dot_acc::BigFloat, dot_tmp::BigFloat)
    for k in 1:n-1
        _bf_dot!(dot_acc, dot_tmp, vs[k], A)
        _bf_axpy_neg!(dot_acc, vs[k], A, dot_tmp)  # dot_tmp safe to reuse here
    end
end

# Allocate BigFloat scratch buffers only when T=BigFloat; returns (nothing, nothing) otherwise.
_alloc_bf_scratch(::Type{BigFloat}) = (BigFloat(0), BigFloat(0))
_alloc_bf_scratch(::Type{T}) where {T} = (nothing, nothing)

# Dispatch: route to zero-alloc MPFR path for BigFloat, standard path otherwise.
_reortho_fro!(vs, n, A, ::Nothing, ::Nothing) = _reortho_fro!(vs, n, A)
_reortho_fro!(vs, n, A, acc::BigFloat, tmp::BigFloat) = _reortho_fro_bf!(vs, n, A, acc, tmp)
_reortho_dro!(vs, n, A, ::Nothing, ::Nothing) = _reortho_dro!(vs, n, A)
function _reortho_dro!(vs, n, A, acc::BigFloat, tmp::BigFloat)
    _reortho_fro_bf!(vs, n, A, acc, tmp)
    _reortho_fro_bf!(vs, n, A, acc, tmp)
end


"""
    _lanczos_vectorial!(E_eff, V_eff, max_iter, epsilon, reortho) -> (b, Kdim, reortho_count)

Core Lanczos iteration in the Liouvillian eigenbasis.  Uses the ghost-b trick:
`b[1] = 1` is a placeholder so that `b[k]` and `vs[k]` share the same index;
the placeholder is stripped before returning.

Supports `:none`, `:fro`, and `:dro` reorthogonalization.
For BigFloat, dispatches to zero-alloc MPFR helpers automatically.

Returns `(b[2:end], Kdim, reortho_count)` where `Kdim` is the converged
Krylov dimension (>0), or -1 if `max_iter` was reached.
"""
function _lanczos_vectorial!(E_eff::Vector{T}, V_eff::Vector{T},
    max_iter::Int,
    epsilon::T,
    reortho::Symbol) where {T<:AbstractFloat}

    dot_acc, dot_tmp = _alloc_bf_scratch(T)

    K_eff = length(E_eff)
    vs = [Vector{T}(undef, K_eff) for _ in 1:max_iter+1]
    b = T[]
    reortho_count = 0
    A    = Vector{T}(undef, K_eff)


    # n=0 step 1 normalize initial vector (should be already the case) O_0
    vs[1] .= V_eff
    vs[1] ./= norm(vs[1])
    push!(b,1)   #artificial for correct indexing caccountign between b and v - need to remove first entry when returning

    # n=1 step 2 first application O_1
    A .= E_eff .* vs[1]  # L(v) = w .* v
    a1 = dot(vs[1],A)

    A .-= a1 .* vs[1]
    b1 = norm(A)
    b1 < epsilon && return b[2:end], 1, 0

    push!(b,b1)
    vs[2] .= A ./ b1

    for n in 3:max_iter+1 
        A  .= E_eff .* vs[n-1]
        an = dot(vs[n-1], A)
        A .-= an .* vs[n-1]
        A .-= b[n-1] .* vs[n-2]

        if reortho == :fro; _reortho_fro!(vs, n, A, dot_acc, dot_tmp); reortho_count += 1
        elseif reortho == :dro; _reortho_dro!(vs, n, A, dot_acc, dot_tmp); reortho_count += 2
        end

        bn = norm(A)

        (n+1) % 100 == 0 && @info "Lanczos step" n=n-1 b=bn oc=reortho_count
        bn < epsilon && return b[2:end], n-1, reortho_count # Kdim = length(b[2:end]) + 1
        push!(b, bn)
        vs[n] .= A ./ bn
    end

    b = b[2:end]
    return b, -1, reortho_count
end



"""
    _lanczos_vectorial_pro!(E_eff, V_eff, max_iter, epsilon, eps_reortho, double_pass)
        -> (b, Kdim, reortho_count)

Partial Reorthogonalization (PRO) Lanczos in the Liouvillian eigenbasis.

Tracks the loss of orthogonality via the W-matrix recurrence (Simon 1984).
Reorthogonalization is triggered when `max|W[m,n]| > eps_reortho`.  When
triggered, both `vs[n-1]` and `vs[n]` are reorthogonalized.  If `double_pass`
is true each reortho sweep is repeated twice (DPRO variant).

The W recurrence is threaded (`Threads.@threads`) over the m-index.
"""
function _lanczos_vectorial_pro!(E_eff::Vector{T}, V_eff::Vector{T},
    max_iter::Int,
    epsilon::T,
    eps_reortho::T,
    double_pass::Bool) where {T<:AbstractFloat}

    eps_M = eps(T)
    dot_acc, dot_tmp = _alloc_bf_scratch(T)
    
    K_eff = length(E_eff)
    vs = [Vector{T}(undef, K_eff) for _ in 1:max_iter+1]
    a = T[]
    b = T[]
    reortho_count = 0
    A       = Vector{T}(undef, K_eff)
    An_1    = Vector{T}(undef, K_eff)


    w_prev  = zeros(T, max_iter + 1) # W[n-2,:]
    w_curr  = zeros(T, max_iter + 1) # W[n,:]
    w_next  = zeros(T, max_iter + 1) # W[n,:]


    # n=0 step 1 normalize initial vector (should be already the case) O_0
    vs[1] .= V_eff
    vs[1] ./= norm(vs[1])
    push!(b,1)   #artificial for correct indexing caccountign between b and v - need to remove first entry when returning

    # n=1 step 2 first application O_1
    A .= E_eff .* vs[1]  # L(v) = w .* v
    a1 = dot(vs[1],A)
    push!(a,a1)

    w_prev[1] = one(T)                      # W(:,1): self-IP
    w_curr[1] = eps_M; w_curr[2] = one(T)  # W(:,2) after first step

    A .-= a1 .* vs[1]
    b1 = norm(A)
    b1 < epsilon && return b[2:end], 1, 0

    push!(b,b1)
    vs[2] .= A ./ b1

    for n in 3:max_iter+1 
        @. A = E_eff * vs[n-1] - b[n-1] * vs[n-2]
        an = dot(vs[n-1], A)   # vs[n-1] orth vs[n-2] cancels the extra term
        A .-= an .* vs[n-1]
        push!(a,an)


        # BEGIN PRO BLOCK
        bk = norm(A)
        bk < epsilon && return b[2:end], n-1, reortho_count # early stop 
        
        fill!(w_next, zero(T))
        w_next[n-1] = eps_M
        w_next[n] = one(T)

        Threads.@threads for m in 2:n-2
            w_tilde = b[m+1] * w_curr[m+1] + (a[m] - a[n-1]) * w_curr[m] +
                      b[m] * w_curr[m-1] - b[n-1] * w_prev[m]
            w_next[m] = w_tilde / bk
        end

        if n>4 && maximum(abs, @view w_next[2:n-2]) > eps_reortho
            @debug "PRO reorthogonalization triggered" n=n eps_reortho=eps_reortho
            
            # recompute O[n-1]
            An_1  .= vs[n-1] .* b[n-1]
            _reortho_fro!(vs, n-1, An_1, dot_acc, dot_tmp)
            reortho_count += 1
            if double_pass
                _reortho_fro!(vs, n-1, An_1, dot_acc, dot_tmp)
                reortho_count += 1
            end
            bn_1 = norm(An_1)
            bn_1 < epsilon && return b[2:end-1], n-2, reortho_count # early stop, previous step should have terminated
            b[end] = bn_1         
            vs[n-1] .= An_1 ./ bn_1

            # recompute O[n]
            @. A = E_eff * vs[n-1] - b[n-1] * vs[n-2]
            an = dot(vs[n-1], A)
            A .-= an .* vs[n-1]

            _reortho_fro!(vs, n, A, dot_acc, dot_tmp)
            reortho_count += 1
            if double_pass
                _reortho_fro!(vs, n, A, dot_acc, dot_tmp)
                reortho_count += 1
            end
            fill!(@view(w_curr[1:n-2]), eps_M)   # vs[n-1] is now clean
            fill!(@view(w_next[1:n-1]), eps_M)   # A is now clean (leave w_next[n]=1)
        end
        w_prev .= w_curr
        w_curr .= w_next

        # END PRO BLOCK

        bn = norm(A)

        (n+1) % 100 == 0 && @info "Lanczos step" n=n-1 b=bn oc=reortho_count
        bn < epsilon && return b[2:end], n-1, reortho_count # Kdim = length(b[2:end]) + 1
        push!(b, bn)
        a[end] = an
        vs[n] .= A ./ bn
    end

    b = b[2:end]
    return b, -1, reortho_count
end



"""
    lanczos(H, O; precision, reortho, remove_degeneracies, ...) -> (b, Kdim, reortho_count)

Compute the Lanczos b-coefficients for the Liouvillian superoperator L = [H, ·]
acting on the operator O.

The algorithm builds the Liouvillian eigenbasis (CER) from H and O, then runs
vectorial Lanczos in that basis at the requested working precision.

Arguments
- `H`, `O`              : sparse Hermitian matrices (any precision)
- `precision`           : working type -- `:f64`, `:double`, `:bigfloat`, or integer bits
- `reortho`             : `:none`, `:fro`, `:dro`, `:pro`, `:dpro`
- `remove_degeneracies` : collapse degenerate frequency blocks (default `true`)
- `max_iter`            : max Lanczos steps (default K_eff + 10)
- `eps_factor`          : convergence tolerance = eps_factor * sqrt(eps(T))
- `tol_factor`          : degeneracy collapse tolerance
- `eps_reortho_factor`  : PRO trigger = factor * sqrt(eps(T)) (default factor = 1)

Returns `(b, Kdim, reortho_count)`.
"""
function lanczos(H::AbstractMatrix, O::AbstractMatrix;
    precision           = :f64,
    reortho::Symbol             = :pro,
    remove_degeneracies::Bool   = true,
    max_iter                    = nothing,
    eps_factor                  = nothing,
    tol_factor                  = nothing,
    eps_reortho_factor          = nothing)
    
    # set up the precision
    T, bits = _precision_type(precision)    # to downcast from build precision to lanczos precision
    eps_conv   = _resolve_eps(eps_factor, T)       # default: sqrt(eps_M)

    # CER
    E_eff, V_eff = build_liouvillian_eigenbasis(H, O; 
                    tol_factor=tol_factor, remove_degeneracies=remove_degeneracies)
    imax = isnothing(max_iter) ? length(E_eff) + 10 : max_iter
    E_t  = T === BigFloat ? setprecision(() -> T.(E_eff), bits) : T.(E_eff)
    V_t  = T === BigFloat ? setprecision(() -> T.(V_eff), bits) : T.(V_eff)


    reortho == :stochastic && error("reortho = :stochastic is not yet implemented")

    _run() = begin
        if reortho ∈ (:pro, :dpro)
            @info "Lanczos (D)PRO vectorial start" K_eff=length(E_eff) T=T bits=bits reortho=reortho max_iter=imax eps_conv=@sprintf("%.2e", Float64(eps_conv))
            eps_reortho = isnothing(eps_reortho_factor) ? sqrt(eps(T)) : T(eps_reortho_factor*sqrt(eps(T)))
            b, Kdim, reortho_count = _lanczos_vectorial_pro!(E_t, V_t, imax, eps_conv, eps_reortho, reortho == :dpro)
        else
            @info "Lanczos vectorial start" K_eff=length(E_eff) T=T bits=bits reortho=reortho max_iter=imax eps_conv=@sprintf("%.2e", Float64(eps_conv))
            b, Kdim, reortho_count = _lanczos_vectorial!(E_t, V_t, imax, eps_conv, reortho)
        end
        @info "Lanczos done" Kdim=Kdim length_b=length(b) b_last=b[end] reortho_count=reortho_count        
        return b, Kdim, reortho_count
    end

    return T === BigFloat ? setprecision(_run, bits) : _run()

end