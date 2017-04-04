% initialize IRT and gpuNUFFT here -- depends on your setup

clear all; clc;
%LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab



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
image = squeeze(raread('../data/shepplogan.ra'));
N = size(image,1);
nro = 2*N;
npe = 2*N;

%% k-space coordinates: linear radial
traj = zeros(3,nro,npe);
r = 2*pi*(0:nro-1)/nro - pi;
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
osf = 2; wg = 2; sw = 1;
Kirt = reshape(traj(1:2,:,:), 2, []).';
st = nufft_init(Kirt, [N N], wg*[2 2], osf*[N N], [N/2 N/2]);


%% Set up gpuNUFFT
Kgn = reshape(traj, 3, []);
G = gpuNUFFT(Kgn/2/pi,w,osf,wg,sw,[N,N],[],true);


%% Everyone degrids
data_irt = nufft(image, st) / nro / npe;
rawrite(single(data_irt), 'sl_data_irt.ra', 1);

data_gn = G*image(:);
rawrite(single(data_gn), 'sl_data_gn.ra', 1);

data_bart = bart('nufft -c', traj*nro/4/pi, image);
rawrite(single(data_bart), 'sl_data_bart.ra', 1);

!./tron -v ../data/shepplogan.ra sl_data_tron.ra 
data_tron = double(squeeze(raread('sl_data_tron.ra')));
data_tron = data_tron(:);

%% Apply SDC and convert to double
data_irt = double(data_irt(:)) .* w(:);
data_gn = double(data_gn(:)) .* w(:);
data_bart = double(data_bart(:)) .* w(:);
data_tron = double(data_tron(:)) .* w(:);

%% Everyone grids everyone else


image_irt_irt = real(nufft_adj(data_irt, st));
image_gn_irt = real(nufft_adj(data_gn .* w(:), st));
image_bart_irt = real(nufft_adj(data_bart .* w(:), st));
image_tron_irt = real(nufft_adj(data_tron .* w(:), st));

image_irt_gn = reshape(real(G'*data_irt), N, N) / wg^3;
image_gn_gn = reshape(real(G'*data_gn), N, N) / wg^3;
image_bart_gn = reshape(real(G'*data_bart), N, N) / wg^3;
image_tron_gn = reshape(real(G'*data_tron), N, N) / wg^3;


image_irt_bart = bart('nufft -a', traj*nro/4/pi, data_irt) / pi;
image_gn_bart = bart('nufft -a', traj*nro/4/pi, data_gn) / pi;
image_bart_bart = bart('nufft -a', traj*nro/4/pi, data_bart) / pi;
image_tron_bart = bart('nufft -a', traj*nro/4/pi, data_tron) / pi;

!./tron -a -v sl_data_irt.ra  sl_irt_tron.ra
!./tron -a -v sl_data_gn_ra   sl_gn_tron.ra
!./tron -a -v sl_data_bart.ra sl_bart_tron.ra
!./tron -a -v sl_data_tron.ra sl_tron_tron.ra
image_irt_tron = raread('sl_irt_tron.ra');
image_gn_tron = raread('sl_gn_tron.ra');
image_bart_tron = raread('sl_bart_tron.ra');
image_tron_tron = raread('sl_tron_tron.ra');



%% Plot everything

x = [image_irt_irt image_gn_irt image_bart_irt image_tron_irt;
    image_irt_gn image_gn_gn image_bart_gn image_tron_gn;
    image_irt_bart image_gn_bart image_bart_bart image_tron_bart;
    image_irt_tron image_gn_tron image_bart_tron image_tron_tron];
figure(1);
imagesc(x);




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

