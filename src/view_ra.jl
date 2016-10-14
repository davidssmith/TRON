
include("mosaic.jl")
using RawArray
using PyPlot


function rescale(x, factor=0.8)
  xmax = maximum(x)
  x[x .> factor*xmax] = factor*xmax
  x = (x - minimum(x)) / (factor*xmax - minimum(x))
  return x
end

img = raread("img_tron.ra")

println(size(img))
n = round(Int,size(img,4)/2)
m = round(Int,size(img,2)/2)
r=abs(img[1,:,:,110:100:end])
s=rotr90(abs(img[1,m,:,:]))

#imshow(abs(mosaic(squeeze(r,4))),interpolation="nearest",cmap="gray")
figure(1)
imshow(rescale(mosaic(r)),interpolation="nearest",cmap="gray")

figure(2)
imshow((s),interpolation="nearest",cmap="gray")
