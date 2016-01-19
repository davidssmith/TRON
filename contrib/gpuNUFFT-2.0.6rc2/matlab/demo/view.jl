using PyPlot

include("mosaic.jl")

f = open("/tmp/img.fld","r")
nchan, nx, ny, nslices, extra = readdlm("/tmp/img.hdr",Int)
nchan = 8
dims = (1, int(nx), int(ny), int(nslices))
b1dims = (int(nchan), int(nx), int(ny), int(nslices))
println("dims: ", dims)
img = read(f, Complex{Float32}, dims)
close(f)
f = open("/tmp/b1.fld","r")
b1 = read(f, Complex{Float32}, b1dims)
close(f)

# SOS coil combination
#img = sqrt(squeeze(sumabs2(img,1),1))
img = squeeze(img, 1)
#img = squeeze(sumabs2(img, 1),1)

rimg = real(img)
iimg = imag(img)
aimg = abs(img)
pimg = angle(img)

x = squeeze(aimg[int(nx/2),:,:],1)'
g = sqrt(fftshift(abs(ifft(img[:,:,int(nslices/4)]))));
#x = squeeze(img[1,:,:],1)'
y =(aimg[:,:,int(nslices/4)])
figure(1)
clf()
#imshow(g, interpolation="nearest", cmap="gray")
imshow(x, interpolation="nearest", cmap="gray", aspect=1146/nslices)
colorbar()
figure(2)
clf()
imshow(y, interpolation="nearest", cmap="gray")
colorbar()
figure(3)
clf()
m = mosaic(permutedims(squeeze(b1[:,:,:,int(nslices/4)],4), [2,3,1]))
imshow(abs(m), interpolation="nearest", cmap="gray")
colorbar()
