function racompare(file0,file1,file2)
d0 = abs(squeeze(raread(file0)));
d1 = abs(squeeze(raread(file1)));
d2 = abs(squeeze(raread(file2)));
subplot(221);
doplot((d1),file1);
subplot(222);
doplot((d2),file2);
subplot(223);
doplot(d1-d0,'diff');
subplot(224);
doplot(d2-d0,'diff');



function doplot(x, name)
imagesc(x);
title(name);
colorbar;
colormap gray;
