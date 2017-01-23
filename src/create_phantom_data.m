% initialize IRT and gpuNUFFT here -- depends on your setup

clear all; clc; clf;
%LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab



%% Paths

% Fessler's IRT
addpath('../contrib/irt');
% gpuNUFFT
addpath('../contrib/gpuNUFFT-2.0.6rc2/gpuNUFFT');
% BART
setenv('TOOLBOX_PATH','/home/dss/git/bart');
addpath('/home/dss/git/bart/matlab');



%%
N = 256;
image = phantom(N);
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
%k = reshape(k,3,[]);omega = 

% figure(1);
% plot(k(1,:),k(2,:),'.');
% title('sample points');

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


%%
osf = 2; wg = 2; sw = 1;
K = reshape(traj(1:2,:,:), 2, []).';
st = nufft_init(K, [N N], wg*[2 2], osf*[N N], [N/2 N/2]);
data = nufft(image, st);
data = reshape(data, nro, npe) / nro / npe;
data2 = zeros(1,1,nro,npe);
data2(1,1,:,:) = data;
rawrite(single(data2), 'sl_data_irt.ra');
clear data2;
data = data .* w;
image_irt_irt = real(nufft_adj(data(:), st));

data_tron = squeeze(raread('sl_data_tron.ra'));
data_tron = data_tron .* w;
image_tron_irt = imag(nufft_adj(data_tron(:), st));


% figure(1);
% subplot(331);
% imagesc(image);
% title('truth');
% colorbar;

subplot(331);
imagesc(image_irt_irt);
colormap(gray);
title('IRT-IRT');
colorbar;

image_irt_tron = abs(squeeze(raread('sl_irt_tron.ra')));
subplot(332);
imagesc(image_irt_tron);
colormap(gray);
title('IRT-TRON');
colorbar;

subplot(334);
imagesc(log(abs(data_tron)));
colormap('default');

image_irt_tron = abs(squeeze(raread('sl_irt_tron.ra')));
subplot(335);
imagesc(image_tron_irt);
colormap(gray);
title('TRON-IRT');
colorbar;

image_tron_tron = abs(squeeze(raread('sl_tron_tron.ra')));
subplot(333)
imagesc(image_tron_tron);
colormap(gray);
title('TRON-TRON');
colorbar;

nrmse_irt_irt = norm(image_irt_irt - image) / N / N;
%nrmse_irt_tron = norm(image_irt_tron - image) / N / N;

fprintf('IRT-IRT      nmrse: %g\n', nrmse_irt_irt);
%fprintf('IRT-TRON     nmrse: %g\n', nrmse_irt_tron);
% IRT nmrse: 0.0010093

%% gpuNUFFT
K = reshape(traj, 3, []);

G = gpuNUFFT(K/2/pi,w,osf,wg,sw,[N,N],[],true);
data = G*image(:);
data = reshape(data, nro, npe);
image_gn = reshape(real(G'*data(:)), N, N) / wg^3;

subplot(223);
imagesc(abs(squeeze(image_gn)))
colorbar;
colormap(gray);
title('gpuNUFFT');

nrmse_gn = norm(image_gn - image) / N / N;
fprintf('gpuNUFFT nmrse: %g\n', nrmse_gn);

%% Use BART

%traj = bart(sprintf('traj -x %d -y %d -r', nro, npe));
data = bart('nufft -c', traj*nro/4/pi, image);
for i = 1:nro
    for j = 1:npe
        data(1,i,j) = data(1,i,j) * w(i,j);
    end
end
image_bart = bart('nufft -a', traj*nro/4/pi, data) / pi;

subplot(224);
imagesc(abs(squeeze(image_bart)))
colorbar;
colormap(gray);
title('BART');

nrmse_bart = norm(image_bart - image) / N / N;
fprintf('BART     nmrse: %g\n', nrmse_bart);

%%


%% run this paper's code
!make
!./rr2d ../data/ex_whole_body.ra
!mv img_tron.ra ../fig/

%% load  results
m_tron = single(abs(squeeze(raread('../fig/img_tron.ra'))));
m_irt = raread('../fig/img_irt.ra');
m_gn = raread('../fig/img_gpunufft.ra');

%%
zview = (floor(nslices/7):45:nslices);
yview = floor(N/2);

figure(1)
x_irt = normalize(m_irt(:,:,zview));
x_gn = normalize(m_gn(:,:,zview));
x_tron = normalize(m_tron(:,:,zview));
for traj = 1:size(x_tron,3)
    x_tron(:,:,traj) = flipud(fliplr(rot90(x_tron(:,:,traj),-1)));
    x_irt(:,:,traj) = flipud(fliplr(x_irt(:,:,traj)));
    x_gn(:,:,traj) = flipud(fliplr(x_gn(:,:,traj)));
end
x_irt = x_irt(:,80:end-100,:);
x_gn = x_gn(:,80:end-100,:);
x_tron = x_tron(:,80:end-100,:);
x = mosaic([x_irt; x_gn; x_tron]).';
%x = x(80:end-100,:);
imshow(x,[0,0.99]);
imwrite(x,'../fig/fig3.png');
imwrite(double(x),'../fig/fig3.tiff');
axis image;
title('axial');
print -deps '../fig/fig3.eps'

figure(2)
y_irt =normalize(fliplr(squeeze(m_irt(:,yview,:)).'));
y_gn = normalize(fliplr(squeeze(m_gn(:,yview,:)).'));
y_tron = normalize(fliplr(squeeze(m_tron(yview+2,:,:)).')); 
y = [y_irt, y_gn, y_tron];
y = y(10:end,:);
y = imresize(y, [950,size(y,2)]);
imshow(y,[0,0.99]);
imwrite(y, '../fig/fig2.png');
imwrite(double(y), '../fig/fig2.tiff');
colormap(gray);
title('coronal');
print -deps '../fig/fig2.eps'

% figure(3)
% plot(1:N,x_gn(:,N/2),'b',1:N,x_this(:,N/2),'m');
% legend('gpuNUFFT','this');
%     
% figure(4)
% plot(1:N,x_gn(N/2,:),'b',1:N,x_this(N/2,:),'m');
% legend('gpuNUFFT','this');

% 
% figure(5);
% imagesc(x_this);
% colormap(gray);