using MAT

include("coilcompress.jl")
include("correct.jl")

datafile = "data/rawdata_phantom_regridding.mat"

println("reading $datafile")
m = matread(datafile)
nudata = m["rawdata"][:,:,:]
nudata = permutedims(nudata, [3,1,2])
println(size(nudata))


#println("prewhitening data")
#prewhiten!(nudata, frc_noise_data)


#@time nudata = coilcompress(nudata, nvirt=6)


#println("unshifting phase")
#data, slopes, intercepts = unshift(data)

#println("zeroing DC phase")
#zerodcphase!(data)


f = open("/tmp/data.fld","w")
write(f, nudata)
close(f)
