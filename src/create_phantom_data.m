% initialize IRT and gpuNUFFT here -- depends on your setup

clear all; clc;
%LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab

rgb = brewermap(51,'RdBu');


%% Paths
addpath(genpath('../contrib/irt'));
addpath(genpath('/home/dss/git/gpuNUFFT'));
setenv('TOOLBOX_PATH','/home/dss/git/bart');
addpath(genpath('/home/dss/git/bart/matlab'));

%% Parameters
codes = { 'irt', 'gn', 'bart', 'tron' };

% tags
IRT = 1;
GN = 2;
BART = 3;
TRON = 4;

%% Read phantom data
image = double(squeeze(raread('../data/shepplogan.ra')));
N = size(image,1);
nro = 2*N;
npe = 2*N;

%% k-space coordinates: linear radial
traj = zeros(3,nro,npe);
r = (0:nro-1)/nro - 0.5;
for pe = 1:npe
    theta = (pe-1)*pi / npe;
    traj(1,:,pe) = r *cos(theta);
    traj(2,:,pe) = r* sin(theta);
end

%% assume Ram-Lak SDC for now
w = zeros(nro,npe);
a = (1-1/npe)*2/nro;
b = 1/npe;
for pe = 1:npe
    for ro = 1:nro
        r = ro - nro/2;
        w(ro,pe) = a*abs(r) + b;
    end
end

% figure(2);
% imagesc(w)
% xlabel('phase encode');
% ylabel('readout');
% title('sample weights');
% colorbar;


%% Set up IRT
osf = 2; wg = 2; 
Kirt = 2*pi*reshape(traj(1:2,:,:), 2, []).';
st = nufft_init(Kirt, [N N], wg*[2 2], osf*[N N], [N/2 N/2]);

%% Set up gpuNUFFT
Kgn = reshape(traj(1:2,:,:), 2, []);
sw = 16;
G = gpuNUFFT(Kgn,w,osf,wg*[1 1],sw,[N,N],[],true);


%% Everyone degrids
data_irt = reshape(nufft(image, st),1,1, nro,npe);
rawrite(single(data_irt), 'sl_data_irt.ra', 1);

tic;
data_gn = N*double(reshape(G*image(:), 1, 1, nro, npe));
toc;
rawrite(single(data_gn), 'sl_data_gn.ra', 1);

data_bart = N*double(reshape(bart('nufft -c', traj*nro/2, image), 1, 1, nro, npe));
rawrite(single(data_bart), 'sl_data_bart.ra', 1);

%!./shepplogan.sh 


%% Plot the data

data_tron = double(raread('sl_data_tron.ra'));

figure(1);
xirt = abs(squeeze(data_irt));

diff = (abs(squeeze(data_tron)) - xirt) / max(abs(squeeze(data_irt)));
subplot(221); imagesc((abs(squeeze(data_irt)))); title('IRT'); colorbar;
subplot(222); imagesc((abs(squeeze(data_gn)))); title('gpuNUFFT'); colorbar;
subplot(223); imagesc(abs(squeeze(data_bart))); title('BART - IRT'); colorbar;
subplot(224); imagesc(abs(squeeze(data_tron))-xirt); title('TRON - IRT');colorbar;
colormap('default')

%figure(6); imagesc(abs(squeeze(data_tron))-xirt); colorbar;

%% Everyone grids everyone else

irt_scale = N*N*osf*osf;
image_irt_irt = (nufft_adj(data_irt(:) .* w(:), st)) / irt_scale;
image_gn_irt = (nufft_adj(data_gn(:) .* w(:), st)) / irt_scale;
image_bart_irt = (nufft_adj(data_bart(:) .* w(:), st)) / irt_scale;
image_tron_irt = (nufft_adj(data_tron(:) .* w(:), st)) / irt_scale;

tic;
gn_scale = N*osf*osf;
image_irt_gn = reshape(G'*data_irt(:), N, N) / gn_scale;
image_gn_gn = reshape(G'*data_gn(:), N, N) / gn_scale;
image_bart_gn = reshape(G'*data_bart(:), N, N) / gn_scale;
image_tron_gn = reshape(G'*data_tron(:), N, N) / gn_scale;
toc;

bart_scale = N*osf*osf;
image_irt_bart = bart('nufft -a', traj*nro/2, reshape(data_irt(:).*w(:),1,nro,npe)) / bart_scale;
image_gn_bart = bart('nufft -a', traj*nro/2, reshape(data_gn(:).*w(:),1,nro,npe)) / bart_scale;
image_bart_bart = bart('nufft -a', traj*nro/2, reshape(data_bart(:).*w(:),1,nro,npe)) / bart_scale;
image_tron_bart = bart('nufft -a', traj*nro/2, reshape(data_tron(:).*w(:),1,nro,npe)) / bart_scale;

% !./tron -a -v sl_data_irt.ra  sl_irt_tron.ra
% !./tron -a -v sl_data_gn_ra   sl_gn_tron.ra
% !./tron -a -v sl_data_bart.ra sl_bart_tron.ra
% !./tron -a -v sl_data_tron.ra sl_tron_tron.ra

%%
image_irt_tron = squeeze(raread('sl_irt_tron.ra'));
image_gn_tron = squeeze(raread('sl_gn_tron.ra'));
image_bart_tron = squeeze(raread('sl_bart_tron.ra'));
image_tron_tron = squeeze(raread('sl_tron_tron.ra'));

figure(2);
subplot(221); rimp(image_irt_irt); title('IRT-IRT');
subplot(222); rimp(image_irt_gn); title('IRT-gpuNUFFT');
subplot(223); rimp(image_bart_bart); title('IRT-BART');
subplot(224); rimp(image_tron_tron); title('IRT-TRON');

figure(3);
subplot(221); rimp(image_gn_irt); title('gpuNUFFT-IRT');
subplot(222); rimp(image_gn_gn); title('gpuNUFFT-gpuNUFFT');
subplot(223); rimp(image_gn_bart); title('gpuNUFFT-BART');
subplot(224); rimp(image_gn_tron); title('gpuNUFFT-TRON');

figure(4);
subplot(221); rimp(image_bart_irt); title('BART-IRT');
subplot(222); rimp(image_bart_gn); title('BART-gpuNUFFT');
subplot(223); rimp(image_bart_bart); title('BART-BART');
subplot(224); rimp(image_bart_tron); title('BART-TRON');

figure(5);
subplot(221); rimp(image_tron_irt); title('TRON-IRT');
subplot(222); rimp(image_tron_gn); title('TRON-gpuNUFFT');
subplot(223); rimp(image_tron_bart); title('TRON-BART');
subplot(224); rimp(image_tron_tron); title('TRON-TRON');

%%



x = lmsediff(image, abs(image_irt_irt));
y = lmsediff(image, abs(image_gn_gn));
z = lmsediff(abs(image_bart_bart), image);
w = lmsediff(abs(image_tron_tron), image);
fprintf('IRT MRMSE:      %g\n', sqrt(mean(x(:).^2)));
fprintf('gpuNUFFT MRMSE: %g\n', sqrt(mean(y(:).^2)));
fprintf('BART MRMSE:     %g\n', sqrt(mean(z(:).^2)));
fprintf('TRON MRMSE:     %g\n', sqrt(mean(w(:).^2)));  % 0.02856

a = -0.25; %min([x(:); y(:); z(:); w(:)]);
b = 0.25; %max([x(:); y(:); z(:); w(:)]);
figure(6);
subplot(221);
imagesc(x); colorbar; title('IRT'); caxis([a,b]);
subplot(222);
imagesc(y); colorbar; title('gpuNUFFT');  caxis([a,b]);
subplot(223);
imagesc(z); colorbar; title('BART'); caxis([a,b]);
subplot(224);
imagesc(w); colorbar; title('TRON'); caxis([a,b]);
colormap(rgb);
%% Plot everything
data_tron = double(raread('sl_data_tron.ra'));
data_irt = double(raread('sl_data_irt.ra'));

%x = [squeeze(data_irt).*w squeeze(data_gn).*w squeeze(data_bart).*w squeeze(data_tron).*w];

xirt = fftshift(fft(fftshift(squeeze(data_irt),1),[],1),1);
xtron = fftshift(fft(fftshift(squeeze(data_tron),1),[],1),1);
xtron = squeeze(data_tron).*w;
%xirt = squeeze(data_irt).*w;
figure(1);

subplot(121);
imagesc((abs(xirt)));   
colorbar;
subplot(122);
imagesc(log(abs(xtron)));
colorbar;
colormap(gray)

%%
I = iradon(abs(squeeze(xirt)),linspace(-180,0*(1-1/npe),npe));
imagesc(I)

%%
figure(1)
imagesc(abs(squeeze(image_tron_irt)));
colormap(gray);

%%
x = [image_irt_irt image_gn_irt image_bart_irt image_tron_irt;
    image_irt_gn image_gn_gn image_bart_gn image_tron_gn;
    image_irt_bart image_gn_bart/nro/npe image_bart_bart/nro/npe image_tron_bart];

y = [image_irt_tron/nro/npe image_gn_tron/nro/npe image_bart_tron/nro/npe image_tron_tron/20/N/N];
figure(2);
imagesc((abs(y)));




%% compute errors



nrmse_irt_irt = norm(image_irt_irt - image) / N / N;
%nrmse_irt_tron = norm(image_irt_tron - image) / N / N;

fprintf('IRT-IRT      nmrse: %g\n', nrmse_irt_irt);
%fprintf('IRT-TRON     nmrse: %g\n', nrmse_irt_tron);
% IRT nmrse: 0.0010093

nrmse_gn = norm(image_gn - image) / N / N;
fprintf('gpuNUFFT nmrse: %g\n', nrmse_gn);


nrmse_bart = norm(image_bart - image) / N / N;
fprintf('BART     nmrse: %g\n', nrmse_bart);


%% Use BART

%traj = bart(sprintf('traj -x %d -y %d -r', nro, npe));


%% run this paper's code
!make
!./rr2d ../data/ex_whole_body.ra
!mv img_tron.ra ../fig/

