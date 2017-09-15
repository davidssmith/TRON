% raview.m
function raview(filename)
d = (squeeze(raread(filename)));
%n = size(d,3);
%d = d(:,:,round(n/3));
%d =mosaic(squeeze(d(1,:,:,:)));
r = real((d));
i = imag((d));
m = (abs((d)));
p = angle((d))/pi;
subplot(221);
doplot(r, 'real');
subplot(222);
doplot(i,'imag');
subplot(223);
doplot(m,'mag');
subplot(224);
doplot(p,'phase');


function doplot(x, name)
imagesc(x);
title(name);
colorbar;
colormap gray;
