

%% Read phantom data
data = double(squeeze(raread('../data/cmt_phantom_la.ra')));
nchan = size(data,1);
nro = size(data,2);
npe = size(data,3);
npe_per_slice = 512;
prof_slide = 512;
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

outfile = 'output/img_laph_irt.ra';
if exist(outfile) == 0
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
          %theta = 111.246*(pe-1+p1-1)*pi / 180;
          theta = 2*pi*(pe-1)/ npe_per_slice + pi/2;
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
  rawrite(img_irt, outfile);
  telapsed_irt = toc(tstart);
end

%% gpuNUFFT grids all

outfile = 'output/img_laph_gn.ra';
if exist(outfile) == 0
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
          theta = 2*pi*(pe-1)/ npe_per_slice + pi/2;
          traj(1,:,pe) = r *cos(theta);
          traj(2,:,pe) = r* sin(theta);
      end
      Kgn = reshape(traj(1:2,:,:), 2, []);
      G = gpuNUFFT(Kgn,w,osf,wg*[1 1],sw,[nx,ny],[],true);
      for c = 1:nchan
          B = data_this_slice(c,:,:);
          tmp = reshape(G'*(B(:).*sqrt(w(:))), nx, ny) / gn_scale;
          img_gn(:,:,z) = img_gn(:,:,z) + abs(tmp).^2;
      end
  end
  img_gn = sqrt(img_gn);
  rawrite(img_gn, outfile);
  telapsed_gn = toc(tstart);
end

%% BART grids all
outfile = 'output/img_laph_bart.ra';
if exist(outfile) == 0
  tstart = tic;
  bart_scale = sqrt(nro*npe_per_slice);
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
      theta = 2*pi*(pe-1)/ npe_per_slice + pi/2;
      traj(1,:,pe) = r *cos(theta);
      traj(2,:,pe) = r* sin(theta);
    end
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
  rawrite(img_bart, outfile);
  telapsed_bart = toc(tstart);

  % save run times
  name = {'IRT';'gpuNUFFT';'BART'};
  cputime = [telapsed_irt; telapsed_gn; telapsed_bart];
  T = table(name, cputime , 'RowNames',name);
  writetable(T,'figs/laph_timings.csv');
end



%% plot run times
T = readtable('figs/laph_timings.csv');
cputime = T.cputime;
cputime(4) = 0.76;
cputime = cputime([4,2,1,3]);
h6 = figure(6);
set(gcf,'color','w');
subplot(122);
barh(cputime,'black'); axis('square');
title('run times (s)');
set(gca,'FontSize',14);
yticklabels({'TRON','gpuNUFFT','IRT','BART'});

text(cputime*1.05,1:4,num2str(cputime,'%4.3g'));




%% reload recons so that can develop without running recons again
% run ./01_run_tron.sh first
img_tron = squeeze(raread('output/img_laph_tron.ra'));
img_irt = squeeze(raread('output/img_laph_irt.ra'));
img_gn = squeeze(raread('output/img_laph_gn.ra'));
img_bart = squeeze(raread('output/img_laph_bart.ra'));



%% Plot recon of phantom data
z = 6; %[10,27,40]; %floor(nz/4);
x = 1;
y = 1;
Xirt = mosaic(img_irt(x:end,y:end,z)); %rot90(img_irt(:,:,z));
Xgn = mosaic(img_gn(x:end,y:end,z)); %rot90(img_gn(:,:,z));
Xtron = mosaic(img_tron(x:end,y:end,z)); %rot90(img_tron(:,:,z));
Xbart = mosaic(img_bart(x:end,y:end,z)); %rot90(img_bart(:,:,z));


fprintf('max(Xirt): %g\nmax(Xgn): %g\nmax(bart): %g\nmax(Xtron): %g\n', ...
    max(Xirt(:)), max(Xgn(:)), max(Xbart(:)),max(Xtron(:)));

X = [Xirt, Xgn; Xbart, Xtron];
a = min(X(:));
b = max(X(:));

subplot(121); imagesc(X, [a,b]);
set(gca,'FontSize',14);

title('linear-angle phantom');
colormap(gray); axis image; axis off;

text(10,20,'IRT','Color','w');
text(nx+10,20,'gpuNUFFT','Color','w');
% text(1024+10,20,'BART','Color','white');
% text(1024+512+10,20,'TRON','Color','white');
text(10,ny,'BART','Color','w');
text(nx+10,ny,'TRON','Color','w');


h6.PaperPositionMode = 'manual';
print(h6, 'figs/fig6','-dpdf','-fillpage');

%imwrite(X, 'figs/fig3.png');

h6.Color = 'white';
set(h6, 'InvertHardCopy', 'off');
orient(h6,'landscape');

print 'figs/fig6' -dpng
print 'figs/fig6' -depsc
print 'figs/fig6' -dpdf



%% Compute SSIMs
%
%ssimval = ssim(A,ref)
%[ssimval,ssimmap] = ssim(A,ref)
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
writetable(T,'figs/laph_ssim.csv');

% gpuNUFFT SSIM: 0.986678
% BART SSIM:     0.988956
% TRON SSIM:     0.996486

