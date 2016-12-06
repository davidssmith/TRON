% initialize IRT and gpuNUFFT here -- depends on your setup

clear all;

%%
N = 128;
image = phantom(N);
nro = 2*N;
npe = 2*N;

%% k-space coordinates: linear radial
k = zeros(2,nro,npe);
r = 2*pi*(0:nro-1)/nro - pi;
for pe = 1:npe
    theta = (pe-1)*pi / npe;
    k(1,:,pe) = r *cos(theta);
    k(2,:,pe) = r* sin(theta);
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

figure(2);
imagesc(w)
xlabel('phase encode');
ylabel('readout');
title('sample weights');
colorbar;

%% Fessler's IRT
addpath('../contrib/irt');


%%
addpath('../contrib/gpuNUFFT-2.0.6rc2/gpuNUFFT');


%%


osf = 2; wg = 2; sw = 1;
K = reshape(k, 2, []).';
st = nufft_init(K, [N N], wg*[2 2], osf*[N N], [N/2 N/2]);
data = nufft(image, st);
data = reshape(data, nro, npe) / nro / npe;
data = data .* w;
image_irt = real(nufft_adj(data(:), st));

figure(3);
imagesc(abs(data));
title('data');

figure(4);
imagesc([image_irt; image]);
colormap(gray);
title('IRT image');
colorbar;

nrmse = norm(image_irt - image) / N / N;

fprintf('IRT nmrse: %g\n', nrmse);
% IRT nmrse: 0.0010093

%% gpuNUFFT

G = gpuNUFFT(K,w,osf,wg,sw,[N,N],[],true);
data = G*image(:);
data = reshape(data, nro, npe) / nro / npe;
data = data .* w;
image_gn = reshape(real(G'*data(:)), N, N);

%%

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
for k = 1:size(x_tron,3)
    x_tron(:,:,k) = flipud(fliplr(rot90(x_tron(:,:,k),-1)));
    x_irt(:,:,k) = flipud(fliplr(x_irt(:,:,k)));
    x_gn(:,:,k) = flipud(fliplr(x_gn(:,:,k)));
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
