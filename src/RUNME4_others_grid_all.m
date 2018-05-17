


%% Everyone grids phantom data

irt_scale = nro*npe;
image_irt_irt = (nufft_adj(data_irt(:) .* w(:), st)) / irt_scale;
rawrite(image_irt_irt, 'output/sl_irt_irt.ra');

gn_scale = nro*npe*sqrt(2)/N;
image_irt_gn = reshape(G'*(data_irt(:).*sqrt(w(:))), N, N) / gn_scale;
rawrite(image_irt_gn, 'output/sl_irt_gn.ra');

bart_scale =  nro*npe/N;
image_irt_bart = bart('nufft -a', traj*nro/2, reshape(data_irt(:).*w(:),1,nro,npe)) / bart_scale;
rawrite(image_irt_bart, 'output/sl_irt_bart.ra');

%%
image_irt_irt = squeeze(raread('output/sl_irt_irt.ra'));
image_irt_gn = squeeze(raread('output/sl_irt_gn.ra'));
image_irt_bart = squeeze(raread('output/sl_irt_bart.ra'));
image_irt_tron = squeeze(raread('output/sl_irt_tron.ra'));


fprintf('IRT max:  %g\n', max(abs(image_irt_irt(:))));
fprintf('TRON max: %g\n', max(abs(image_irt_tron(:))));


%%

u = 1;
l = 0;


h2 = figure(2);
clf;
x = [rimp(image_irt_irt, image); 
    rimp(image_irt_gn, image); 
    rimp(image_irt_bart, image); 
    rimp(image_irt_tron, image)];
imshow(x,[l u]); colorbar;
imwrite(x, 'figs/fig2.png');
%savefig(h2, 'fig2.pdf','compact'); 
text(10,20,'IRT','Color','white');
text(10,262,'gpuNUFFT','Color','white');
text(10,532,'BART','Color','white');
text(10,788,'TRON','Color','white');
text(10,1004,'real','Color','white');
text(266,1004,'imag','Color','white');
text(532,1004,'mag','Color','white');
text(778,1004,'phase','Color','white');
text(1034,1004,'diff','Color','white');

h2.PaperPositionMode = 'manual';
orient(h2,'landscape')
set(h2, 'InvertHardCopy', 'off');
h2.Color = 'white';
print(h2, 'figs/fig2','-dpdf','-fillpage')
print(h2, 'figs/fig2','-deps')
print(h2, 'figs/fig2','-dpng')

rmse_irt = rmse(image_irt_irt, image);
rmse_gn = rmse(image_irt_gn, image);
rmse_bart = rmse(image_irt_bart, image);
rmse_tron = rmse(image_irt_tron, image);

sl_errors = {'IRT', rmse_irt;  'gpuNUFFT', rmse_gn; 'BART',rmse_bart; 'TRON', rmse_tron}
csvwrite('figs/sl_rmse.csv',[rmse_irt, rmse_gn, rmse_bart, rmse_tron]);


%% Read whole body data
data = double(squeeze(raread('../data/ex_whole_body.ra')));
% radims: 6 512 20271
nchan = size(data,1);
nro = size(data,2);
npe = size(data,3);
npe_per_slice = 204;
prof_slide = 21;
nx = nro / 2;
ny = nx;
%nz =  floor(npe / prof_slide) - 1;
nz = floor(1 + (npe - npe_per_slice) / prof_slide);


traj = zeros(3,nro,npe_per_slice);
w = zeros(nro,npe_per_slice);
a = (1-1/npe_per_slice)*2/nro;
b = 1/npe_per_slice;
for pe = 1:npe_per_slice
    for ro = 1:nro
        w(ro,pe) = a*abs(ro - nro/2) + b;
    end
end
tmp = zeros(nx, ny);

osf = 2; 
wg = 2;
sw = 16;


%% IRT grids all
tstart = tic;

irt_scale = nro*npe_per_slice;

img_irt = zeros(nx, ny, nz);
r = (0:nro-1)/nro - 0.5;
for z = 1:nz
    p1 = (z-1)*prof_slide + 1;
    p2 = p1 + npe_per_slice - 1;
    fprintf('IRT slice %d/%d pe:%d-%d\n', z, nz, p1, p2);
    data_this_slice = data(:,:,p1:p2);

    for pe = 1:npe_per_slice
        theta = 111.246*(pe-1+p1-1)*pi / 180;
        traj(1,:,pe) = r *cos(theta);
        traj(2,:,pe) = r* sin(theta);
    end
    Kirt = 2*pi*reshape(traj(1:2,:,:), 2, []).';
    st = nufft_init(Kirt, [nx ny], wg*[2 2], osf*[nx ny], [nx/2 ny/2]);
    for c = 1:nchan
        B = data_this_slice(c,:,:);
        tmp = nufft_adj(B(:) .* w(:), st) / irt_scale;
        img_irt(:,:,z) = img_irt(:,:,z) + abs(tmp).^2;
    end
end
img_irt = sqrt(img_irt);
rawrite(img_irt, 'output/img_cmt_irt.ra');
telapsed_irt = toc(tstart);

%% gpuNUFFT grids all
 
tstart = tic;

gn_scale = nro*npe_per_slice*sqrt(2)/nx;
img_gn = zeros(nx, ny, nz);
r = (0:nro-1)/nro - 0.5;
for z = 1:nz
    p1 = (z-1)*prof_slide + 1;
    p2 = p1 + npe_per_slice - 1;
    fprintf('gpuNUFFT slice %d/%d pe:%d-%d\n', z, nz, p1, p2);
    data_this_slice = data(:,:,p1:p2);    
    for pe = 1:npe_per_slice
        theta = 111.246*(pe-1+p1-1)*pi / 180;
        traj(1,:,pe) = r *cos(theta);
        traj(2,:,pe) = r* sin(theta);
    end
    Kgn = reshape(traj(1:2,:,:), 2, []);
    G = gpuNUFFT(Kgn,w,osf,wg*[1 1],sw,[nx,ny],[],true);
    for c = 1:nchan
        B = data_this_slice(c,:,:);
        tmp = reshape(G'*(B(:).*sqrt(w(:))), nx, ny) / gn_scale;
        img_gn(:,:,z) = img_gn(:,:,z) + abs(tmp).^2 ;
    end
end
img_gn = sqrt(img_gn);
rawrite(img_gn, 'output/img_cmt_gn.ra');
telapsed_gn = toc(tstart);


%% BART grids all
 
tstart = tic;

%bart_scale = nro*npe_per_slice/nx;
bart_scale = nx/sqrt(2);
nx1 = round(nx/2)+1;
nx2 = round(3*nx/2);
ny1 = round(ny/2)+1;
ny2 = round(3*ny/2);
img_bart = zeros(nx, ny, nz);
r = (0:nro-1) - 0.5*nro;
for z = 1:nz
    p1 = (z-1)*prof_slide + 1;
    p2 = p1 + npe_per_slice - 1;
    fprintf('BART slice %d/%d pe:%d-%d\n', z, nz, p1, p2);
    data_this_slice = single(data(:,:,p1:p2));
    for pe = 1:npe_per_slice
        theta = 111.246*(pe-1+p1-1)*pi / 180;
        traj(1,:,pe) = r *cos(theta);
        traj(2,:,pe) = r* sin(theta);
    end
%     for c = 1:nchan
%         B = squeeze(data_this_slice(c,:,:));
%         tmp = bart('nufft -a', traj, reshape(B(:).*w(:),1,nro,npe_per_slice)) / bart_scale;
%         % crop tmp
%         img_bart(:,:,z) = img_bart(:,:,z) + abs(tmp(nx1:nx2,ny1:ny2)).^2;
%     end
     B = squeeze(data_this_slice);
     for c = 1:nchan
         B(c,:,:) = squeeze(B(c,:,:)) .* w(:,:);
     end
     B = reshape(permute(B, [2,3,1]), 1, nro, npe_per_slice, nchan);
     tmp = bart('nufft -a  ', traj, B) / bart_scale;
     for c = 1:nchan
         img_bart(:,:,z) = img_bart(:,:,z) + squeeze(abs(tmp(nx1:nx2,ny1:ny2,c)).^2);
     end
end
img_bart = sqrt(img_bart);
rawrite(img_bart,'output/img_cmt_bart.ra');
telapsed_bart = toc(tstart);

%% save run times

name = {'IRT';'gpuNUFFT';'BART'};
cputime = [telapsed_irt; telapsed_gn; telapsed_bart];
T = table(name, cputime , 'RowNames',name);
writetable(T,'figs/cmt_timings.csv');



%% plot run times
T = readtable('figs/cmt_timings.csv');
cputime = T.cputime;
cputime(4) = 3.28;
cputime = cputime([4,2,1,3]);
h4 = figure(4);
barh(cputime,'black');
xlabel('run time for whole-body data set (s)');
set(gca, 'Xscale','log','FontSize',14);
yticklabels({'TRON','gpuNUFFT','IRT','BART'});
h4.PaperPositionMode = 'manual';
text(cputime*1.05,1:4,num2str(cputime,'%4.3g'));
orient(h4,'landscape');
print(h4, 'figs/fig4','-dpdf','-fillpage');


%% reload recons so that can develop without running recons again
% run ./01_run_tron.sh first
img_tron = squeeze(raread('output/img_cmt_tron.ra'));
img_irt = squeeze(raread('output/img_cmt_irt.ra'));
img_gn = squeeze(raread('output/img_cmt_gn.ra'));
img_bart = squeeze(raread('output/img_cmt_bart.ra'));


%% Plot recon of whole-body data

Xirt = whole_body_mosaic(img_irt);
Xgn = whole_body_mosaic(img_gn);
Xtron = whole_body_mosaic(img_tron);
Xbart = whole_body_mosaic(img_bart);


fprintf('max(Xirt): %g\nmax(Xgn): %g\nmax(bart): %g\nmax(Xtron): %g\n', ...
    max(Xirt(:)), max(Xgn(:)), max(Xbart(:)),max(Xtron(:)));

X = [Xirt, Xgn; Xbart, Xtron];
a = min(X(:));
b = max(X(:));

h3 = figure(3); clf; imagesc(X, [a,0.9*b]); colormap(gray); axis image; axis off;

text(10,20,'IRT','Color','w');
text(512+10,20,'gpuNUFFT','Color','w');
% text(1024+10,20,'BART','Color','white');
% text(1024+512+10,20,'TRON','Color','white');
text(10,512,'BART','Color','w');
text(512+10,512,'TRON','Color','w');



%imwrite(X, 'figs/fig3.png');

h3.PaperPositionMode = 'manual';
h3.Color = 'white';
set(h3, 'InvertHardCopy', 'off');
orient(h3,'landscape');

print 'figs/fig3' -dpng
print 'figs/fig3' -deps
print 'figs/fig3' -dpdf



%% Compute SSIMs
%
%ssimval = ssim(A,ref)
%[ssimval,ssimmap] = ssim(A,ref)
z = 350;
x = real(img_irt(:,:,z));
[ssim_gn, ssim_map_gn] = ssim(double(real(img_gn(:,:,z))), x);
[ssim_bart, ssim_map_bart] = ssim(double(real(img_bart(:,:,z))), x);
[ssim_tron, ssim_map_tron] = ssim(double(real(img_tron(:,:,z))), x);

% figure(6); clf;
% subplot(131); imagesc(ssim_map_gn); xlabel('gpuNUFFT SSIM map'); colorbar; caxis([-1,1]);
% subplot(132); imagesc(ssim_map_bart); xlabel('BART SSIM map'); colorbar; caxis([-1,1]);
% subplot(133); imagesc(ssim_map_tron); xlabel('TRON SSIM map'); colorbar; caxis([-1,1]);
% rgb = brewermap(101,'RdBu')
% colormap(rgb);

fprintf('gpuNUFFT SSIM: %g\n', ssim_gn);
fprintf('BART SSIM:     %g\n', ssim_bart);
fprintf('TRON SSIM:     %g\n', ssim_tron);


name = {'gpuNUFFT';'BART';'TRON'};
ssims = [ssim_gn; ssim_bart; ssim_tron];
T = table(name, ssims , 'RowNames',name);
writetable(T,'figs/cmt_ssim.csv');

% gpuNUFFT SSIM: 0.986678
% BART SSIM:     0.988956
% TRON SSIM:     0.996486

