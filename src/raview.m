% raview.m
function raview(filename)
d = raread(filename);
r = real(squeeze(d));
i = imag(squeeze(d));
m = (abs(squeeze(d)));
p = angle(squeeze(d))/pi;
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
