
include("mosaic.jl")
using RawArray
using PyPlot


r = raread("img_tron.ra")

println(size(r))
n = round(Int,size(r,4)/2)
r=abs(r[1,:,:,n])
println(size(r))

#imshow(abs(mosaic(squeeze(r,4))),interpolation="nearest",cmap="gray")
imshow(r,interpolation="nearest",cmap="spectral")
