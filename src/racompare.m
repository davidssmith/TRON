function racompare(file1,file2)
d1 = abs(squeeze(raread(file1)));
d2 = abs(squeeze(raread(file2)));
subplot(131);
doplot(d1, file1);
subplot(132);
doplot(d2,file2);
subplot(133);
doplot(d1-d2,'diff');



function doplot(x, name)
imagesc(x);
title(name);
colorbar;
colormap gray;
