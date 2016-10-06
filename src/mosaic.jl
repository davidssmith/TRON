function mosaic{T,N}(img::Array{T,N})
  # Create a 2-D mosaic of images from an n-D stack.
  # The first two dimensons are preserved and all additional
  # dimensions are lumped together as tiles
  imsize = size(img)
  n1 = imsize[1]
  n2 = imsize[2]
  n3 = prod(imsize[3:end])
  img = reshape(img, n1, n2, n3)
  n = round(Int, round(sqrt(n3)))  # starting point: integer nearest to the square root
  # find largest integer less than or equal to sqrt that evenly divides the
  # number of images in the stack
  k = 0
  for k in n:-1:1
    if n3 % k == 0
      break
    end
  end
  # figure out the best dimensions
  j = round(Int, n3 / k)
  ny = round(Int, min(j, k))
  nx = round(Int, max(j, k))
  # stick them together
  M = zeros(eltype(img), ny*n1, nx*n2)
  for k in 1:ny, j in 1:nx
    M[1 + n1*(k-1):n1*k, 1 + n2*(j-1):n2*j] = img[:, :, j+(k-1)*nx]
  end
  M
end
