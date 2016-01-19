
function coilcompress{T}(x::Array{T,3}; nvirt=6)
  # Geometric coil compression with alignment to middle
  println("compressing data to $nvirt virtual coils")
  nchan, nro, npe = size(x)
  y = Array(T, nvirt, nro, npe)
  n = div(npe,2)  # middle profile, the starting point
  U0 = Array(T, nchan, nvirt)
  Uprev = Array(T, nchan, nvirt)
  for p in n
    t = squeeze(x[:,:,p], 3)
    C = t*t'
    U, s, V = svd(C)
    U0[:] = U[:,1:nvirt]
    y[:,:,p] = U0'*t
  end
  Uprev[:] = U0
  for p in n+1:npe
    t = squeeze(x[:,:,p], 3)
    C = t*t'
    U, s, V = svd(C)
    U = U[:,1:nvirt]
    W, s, V = svd(U'*Uprev)
    U = U*W*V'  # rotate U to align with previous slice
    y[:,:,p] = U'*t
    #Uprev[:] = U  # took this out bc seems unnecessary
  end
  Uprev[:] = U0
  for p in n-1:-1:1
    t = squeeze(x[:,:,p], 3)
    C = t*t'
    U, s, V = svd(C)
    U = U[:,1:nvirt]
    W, s, V = svd(U'*Uprev)
    U = U*W*V' # rotate U to align with previous slice
    y[:,:,p] = U'*t
    #Uprev[:] = U # took this out bc seems unnecessary
  end
  y
end
