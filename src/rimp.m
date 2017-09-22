function rimp(image)
image = squeeze(image);
r = real(image);
i = imag(image);
m = abs(image);
p = angle(image)/pi;
rgb = brewermap(51,'RdGy');
imagesc(abs([r i; m p])); colorbar; colormap('default');
