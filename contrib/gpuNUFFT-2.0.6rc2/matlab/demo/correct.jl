
using Base.Test

include("fitting.jl")

function zerodcphase!{T}(data::Array{Complex{T},3})
  nchan, nro, npe = size(data)
  mid = ifloor(nro/2) + 1
  for c in 1:nchan, p in 1:npe
    ϕ = angle(data[c, mid, p])
    data[c,:,p] *= exp(-1im*ϕ)
  end
  data
end

function test_zerodcphase!(nro=32, npe=4)
   print("testing zerodcphase! on 1 x $nro x $npe data ... ")
   nchan = 1
   ϕ = Float64[0:nro-1] - floor(nro/2)   # midpoint at int(nro/2)+1
   mid = ifloor(nro/2) + 1
   Δ = rand(npe) # each line gets a different constant shift
   f = Array(Complex128, nchan, nro, npe)
   g = Array(Complex128, nchan, nro, npe)
   for p in 1:npe
     f[1,:,p] = rand(nro).*exp(1im*ϕ)
     g[1,:,p] = f[1,:,p]*exp(1im*Δ[p])
   end
   zerodcphase!(g)
   for p in 1:npe, r in 1:nro
     @test_approx_eq f[1,r,p] g[1,r,p]
   end
   println("pass")
end




function unshift{T}(data::Array{Complex{T},3}; golden=false, ncalib=32, verbose=false)
  # data assumed to be nchan x nro x npe
  # K assumed to be ndim x nro x npe

  if verbose
    println("correcting for radial shifts")
  end
  # calculate sinogram
  sino = fftshift(ifft(fftshift(data, 2), 2), 2)
  nchan, nro, npe = size(data)
  data_shifted = Array(Complex{T}, nchan, nro, npe)
  slopes = zeros(nchan, npe)
  intercepts = zeros(nchan, npe)
  Δk = golden ? 21 : 1
  @assert Δk < npe "not enough projections for meaningful correction"
  if verbose
    println("using Δk of ±$(Δk)")
  end
  ncalib = min(ncalib, div(nro,2)-1)
  roidxs = [div(nro,2)-ncalib:div(nro,2)+ncalib]
  r = [-int(nro/2): int(nro/2)-1]
  for p in 1:npe, j in 1:nchan
    B = vec(data[j,roidxs,p])
    b = vec(sino[j,roidxs,p])
    if golden
      a = p > Δk        ? sino[j,roidxs,p-Δk] : nothing
      c = p <= npe - Δk ? sino[j,roidxs,p+Δk] : nothing
    else  # linear
      a = p == 1   ? sino[j,roidxs,npe] : sino[j,roidxs,p-1]
      c = p == npe ? sino[j,roidxs,1]   : sino[j,roidxs,p+1]
    end
    # average spatial projections a and c
    if a == nothing
      a = c
      ac = c
    elseif c == nothing
      c = a
      ac = a
    else
      ac = 0.5*(a + c)
    end
    a = vec(a)
    b = vec(b)
    c = vec(c)
    ac = vec(ac)

    # create a mask for the spatial domain projections
    abcmag = abs(a) + abs(b) + abs(c)
    abcmag = (abcmag - minimum(abcmag)) / (maximum(abcmag) - minimum(abcmag))
    #abcmag[abcmag .< 0.25] = 0.0  # TODO threshold_otsu(abcmag)

    # calculate masked phase difference
    Δφ = golden ? angle(ac[end:-1:1] ./ b) : angle(ac ./ b)
    Δφ = Δφ .* abcmag

    # perform a weighted least squares fit
    W = diagm(abcmag[:])
    M = hcat(r[roidxs], ones(length(roidxs)))
    u = M'*W*M \ M'*W*Δφ
    slopes[j,p] = u[1]
    intercepts[j,p] = u[2]

    # apply phase compensation
    phase_ramp = 0.5*(u[1]*r + u[2])
    b_phase_compensated = vec(sino[j,:,p]) .* exp(1im*phase_ramp)
    B_shifted = fftshift(fft(fftshift(b_phase_compensated)))
    data_shifted[j,:,p] = B_shifted[:]
  end
  (data_shifted, slopes, intercepts)
end


function test_unshift(nro=32, npe=32)
  print("testing unshift() on 1 x $nro x $npe data ... ")
  nchan = 1
  ϕ = Float64[0:nro-1] - floor(nro/2)   # midpoint at int(nro/2)+1
  mid = ifloor(nro/2) + 1
  Δ = rand(npe) # each line gets a different constant shift
  f = Array(Complex128, nchan, nro, npe)
  f[1,:,:] = repmat(rand(nro).*exp(1im*ϕ), 1, npe)
  g, m, b = unshift(f)
  for p in 1:npe
    @test_approx_eq_eps m[p] 0.0 eps(Float64)
    @test_approx_eq_eps b[p] 0.0 eps(Float64)
    for r in 1:nro
      @test_approx_eq f[1,r,p] g[1,r,p]
    end
  end
  println("pass")
end


function delaycorrect{T,N}(data::Array{Complex{T},3}, K::Array{T,N}; validate=false)
    # radial gradient delay correction

    outdir = "validation"
    if validate
        println("validation enabled: run time will be much longer")
        isdir(outdir) || mkdir(outdir)
    end

    nrep, nro, npe = size(data)
    ndim = ndims(data)

    nparam = size(K,1)  # number of gradient channels
    @assert nparam == 2  "only 2D implemented for now"
    println("npe: $npe\nnro: $nro\nndim: $ndim")
    println("fitting $nparam parameters")

    # Column $k$ of the $U$ matrix contains a unit vector that points along
    # the $k$-th profile This matrix is then squared on an element-by-element
    # basis to yield the relative contributions of each gradient axis to the
    # delay of the profile parallel to the readout direction.

    U = zeros(nparam, npe)   # unit vectors pointing along each radial profile in readout direction
    angles = ((π / φ) * [0:npe-1]) % 2π
    for i in 1:npe
        # spherical polar angles corresponding to radial readout orientations
        θ = (π / φ) * (i - 1)
        U[1,i] = cos(θ)
        U[2,i] = sin(θ)
    end
    U = U.^2

    # The uncorrected r coordinates
    r = [0:nro-1]/nro - 0.5

    Δp = int(npe / 100)  # number of profiles to skip
    println("taking every $(Δp)-th profile")

    shape{S}(x::Vector{S}, p::Vector{S}) = p[1]*abs(sinc(p[2]*(x - p[3]))) + p[4]
    #shape{S}(x::Vector{S}, p::Vector{S}) = p[1] ./ (p[2] + abs(x - p[3]).^p[4])

    datamin, datamax = extrema(abs(data[1,:,:]))
    p0 = [datamax,1.0, 0.0, 1.5]
    yfit = similar(data)
    params, resid = nlsfit(shape, float64(abs(squeeze(data[1,:,:],1))), [1:Δp:npe], r, p0)
    rcenter = float32(squeeze(params[3,:],1))
    println("residual $(norm(resid))")
    for i in 1:Δp:npe
        yfit[1,:,i] = shape(r, params[:,i])
        if validate   # plot profiles and their fits
             clf()
             plot(r,abs(data[1,i,:]),"b.", r, abs(yfit[:,i]), "g-")
             ylim(datamin, datamax)
             xlabel("spatial frequency")
             ylabel("power")
             savefig("$outdir/fit$i.png")
        end
    end
    if false
        clf()
        plot(angles,rcenter,"o")
        ylabel("fitting center freq")
        xlabel("angle (rad)")
        savefig("$outdir/rcenter.png")
        clf()
        plot([1:npe],vec(params[4,:]),"o")
        ylabel("profile fit power law index")
        xlabel("profile")
        savefig("$outdir/powerindex.png")
    end

    # Finally, we formulate and solve the linear inverse problem $U^2 \delta =
    # \Delta r_{||} $ , where $d$ are the deduced three-axis gradient delays,
    # oriented along the coordinate axes, $\Delta r_{||}$ is the vector of center
    # offsets calculated from the Cauchy distribution fits, and $U$ is the matrix of
    # unit vectors pointing along the radial profiles.  Matrix $U$ is squared
    # because one factor computes the relative contribution from each axis, and the
    # other factor accounts for the portion of the coordinate offset that is
    # parallel to the readout direction, which is the only information available to
    # us.

    delta = U[:,1:Δp:end]' \ rcenter[1:Δp:end]   # <-- THE MAGIC
    println("derived delays: $delta")

    # These three values in delta should be the $x$, $y$, and $z$ gradient delays,
    # respectively, in units of spatial frequency. Since this is a massively
    # overdetermined problem, we have the luxury of re-computing these from subsets
    # of the full data set.  One prudent approach would be to solve the inverse
    # problem on a sliding window to examine temporal consistency.  Since we only
    # need at minimum three profiles to invert to obtain the three gradient delays,
    # we will set our sliding window width to 3.

    # To correct $k$-space, we simply apply the calculated delays from $\delta$ in
    # the appropriate direction for each $(k_x,k_y,k_z)$ triplet.
    for j in 1:nro, i in 1:npe
      K[1,j,i] += U[1,i] * delta[1]
      K[2,j,i] += U[2,i] * delta[2]
    end
    K = clamp(K, -0.5, 0.5)
    convert(Array{T,3}, K)
end



function window!{T}(x::Vector{T})
  n = length(x)
  x = x .* sinc([0:n-1]/n)  # Lanczos factors
end
function window!{T}(x::Array{T,2})
  m, n = size(x)
  x = x .* window(m,n)
end
function window(m::Int, n::Int)
  nx = int(m/2)
  ny = int(n/2)
  fx = sinc([0:nx-1]/nx)
  fy = sinc([0:ny-1]/ny)
  vcat(fx[end:-1:1], fx) * vcat(fy[end:-1:1], fy)'
end


function prewhiten!{T,N}(data::Array{T,N}, noise::Array{T,2})
  # Prewhitens data by removing noise correlation between coils.
  # Assumes:
  #   channel dimensions is the first dimension of 'data'
  #   noise is an nchannel x nsamples array of the same type as 'data'
  insize = size(data) # save input size for reshaping upon output
  nsamp = size(noise, 2)
  μ = mean(noise, 2)
  σ = std(noise, 2)
  for j in 1:nsamp  # for each noise sample
    noise[:,j] -= μ # zero mean
    noise[:,j] ./= σ #  unit std dev
  end
  C = noise*noise' / (nsamp-1)  # C = noise covariance matrix
  L = chol(C, :L)  # Cholesky decomposition => lower triangular result
  data = full(inv(L)) * data[:,:] # decorrelate the data using the noise correlations
       # ^ full() needed because * not yet defined for Triangular and Complex arrays
  reshape(data, insize)
end

function test_prewhiten!(m=32, n=16)
  srand(1337)
  x = 1000*rand(n,m)+1000im*rand(n,m)
  y = copy(x)
  prewhiten!(y, x)
  μ = mean(y,2)
  σ = std(y,2)
  for j in 1:m
    y[:,j] -= μ
    y[:,j] ./= σ
  end
  Σ = abs(y*y') / (m-1)
  @test_approx_eq_eps Σ eye(n) 1e-1
end


function runtests()
  test_zerodcphase!()
  test_unshift()
  test_prewhiten()
end
