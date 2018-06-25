function imcombined = whole_body_mosaic(img)
img = squeeze(img);
zlist = [400]; %[110,350,400,610,800,900];
xyz = squeeze(img(:,:,zlist));
for k = 1:length(zlist)
   xyz(:,:,k) = fliplr(rot90(xyz(:,:,k))); 
end
%xyz = xyz(32:192,:,:);
xyz = mosaic(xyz,1);
%w = rot90(real(squeeze(img_tron(1,1,:,128,61:end))),-1);
%u =  rot90(real(squeeze(img_tron(1,1,88,50:end-20,61:end))),-1);
w = rot90(real(squeeze(img(:,128,61:end))),-1);
u =  rot90(real(squeeze(img(88,:,61:end))),-1);
wu = [w u];
%imcombined = [xyz; wu];
imcombined = [xyz];
%imcombined = (imcombined - min(imcombined(:))) / (max(imcombined(:)) - min(imcombined(:)));

