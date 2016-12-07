
using RawArray
using PyPlot

function rescale(x, factor=0.8)
  xmax = maximum(x)
  x[x .> factor*xmax] = factor*xmax
  x = (x - minimum(x)) / (factor*xmax - minimum(x))
  return x
end

img = raread("shepplogan_tron.ra")
r = img[1,:,:,1]
figure(1)
imshow(rescale(abs(r)),interpolation="nearest",cmap="gray")
