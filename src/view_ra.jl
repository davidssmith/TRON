
include("mosaic.jl")
include("ra.jl")
using RA
using PyPlot


r = raread("img_tron.ra")

println(size(r))


imshow(abs(mosaic(squeeze(r,1))),interpolation="nearest",cmap="gray")


