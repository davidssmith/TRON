% initialize IRT and gpuNUFFT here -- depends on your setup

clear all; clc; clf;
%LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab


%% 
% Paths
%
% CUSTOMIZE THESE FOR YOUR SETUP!
%
USERHOME=getenv('HOME');
PATH_TO_IRT=genpath('../contrib/irt');
PATH_TO_GPUNUFFT=genpath(sprintf('%s%s', USERHOME, '/git/gpuNUFFT'));
PATH_TO_BART=genpath(sprintf('%s%s', USERHOME, '/git/bart/matlab'));

addpath(PATH_TO_IRT);
addpath(PATH_TO_GPUNUFFT);
setenv('TOOLBOX_PATH',sprintf('%s%s', USERHOME, '/git/bart'));
addpath(PATH_TO_BART);

%% Read phantom data
image = double(squeeze(raread('../data/shepplogan.ra')));
N = size(image,1);
osf = 2;
nro = osf*N;
npe = osf*N;

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
wg = 2;
Kirt = 2*pi*reshape(traj(1:2,:,:), 2, []).';
st = nufft_init(Kirt, [N N], wg*[2 2], osf*[N N], [N/2 N/2]);

%% Set up gpuNUFFT
Kgn = reshape(traj(1:2,:,:), 2, []);
sw = 16;
G = gpuNUFFT(Kgn,w,osf,wg*[1 1],sw,[N,N],[],true);


%% Everyone degrids
data_irt = reshape(nufft(image, st),1,1, nro,npe);
rawrite(single(data_irt), 'output/sl_data_irt.ra', 1);

%tic;
%data_gn = N*double(reshape(G*image(:), 1, 1, nro, npe));
%toc;
%rawrite(single(data_gn), 'output/sl_data_gn.ra', 1);

%data_bart = N*double(reshape(bart('nufft -c', traj*nro/2, image), 1, 1, nro, npe));
% rawrite(single(data_bart), 'sl_data_bart.ra', 1);

data_tron = double(raread('output/sl_data_tron.ra'));

%!./shepplogan.sh 


%% Plot the data

h = figure(1);
clf;
diff = (squeeze(abs(data_tron)-abs(data_irt))) ./ max(abs(squeeze(data_irt))); 
subplot(221); imshow(log(abs(squeeze(data_irt))),[-6,10]); title('IRT k-space'); colorbar; axis('off');
subplot(222); imshow(log(abs(squeeze(data_tron))),[-6,10]); title('TRON k-space');colorbar; axis('off');
subplot(223); imshow(diff,[-4e-4,4e-4]); title('IRT - TRON');colorbar;
axis('off');

colormap(gray)
fprintf('Data NMSE: %g\n', norm(data_irt(:) - data_tron(:)) / max(abs(data_irt(:))));
%figure(6); imagesc(abs(squeeze(data_tron))-xirt); colorbar;

% Plot a single radial profile between IRT and TRON
a = 1;
b = 2*N;
c = N ; %round(N/2);
kk = (1:length(a:b))/N/2 - 0.5;
x = abs(squeeze(data_irt(1,1,a:b,c)));
y = abs(squeeze(data_tron(1,1,a:b,c)));
figure(1);
subplot(224);
semilogy(kk,x,'k-', kk,y,'k.');
legend('IRT','TRON');
xlabel('frequency','fontsize',12);
ylabel('coefficient magnitude','fontsize',12);
title('profile trace, column 256')
%fixfig;
%savefig('fig1.pdf','compact');
h.PaperPositionMode = 'manual';
orient(h,'landscape')
print(h,'figs/fig1','-dpdf','-fillpage');
