% initialize IRT and gpuNUFFT here -- depends on your setup

clear all;
addpath('../contrib/gpuNUFFT-2.0.6rc2/gpuNUFFT');

%%
data = single(squeeze(raread('../data/ex_whole_body.ra')));
[nchan,nro,npe]=size(data);
data = shiftdim(data,1);

%%
N=3*nro/4;
npe_per_slice = nro/2;
dpe = 89;

nslices = floor((npe - npe_per_slice) / dpe);
disp_slice=round(nslices/2);

%% k-space coordinates: golden angle radial
k = zeros(2,nro,npe);
r = 2*pi*(0:nro-1)/nro - pi;
for pe = 1:npe
    theta = 111.246*(pe-1)*pi / 180;
    k(1,:,pe) = r *cos(theta);
    k(2,:,pe) = r* sin(theta);
end
%k = reshape(k,3,[]);omega = 

%% assume Ram-Lak SDC for now
w = zeros(nro,npe);
for pe = 1:npe
    for ro = 1:nro
        w(ro,pe) = 2*(abs(ro - nro/2) + 1) / (nro+1);
    end
end

%% gpuNUFFT
osf = 2; wg = 2; sw = 1;
img = zeros(N,N,nslices,nchan,'single');
tic;
for z = 1:nslices
    fprintf('gn z %d\n', z);
    pe1 = (z-1)*dpe + 1;
    pe2 = pe1 + npe_per_slice-1;
    tmpdata = reshape(data(:,pe1:pe2,:),[],nchan);
    %tmpdata(:,:) = 1.;
    tmpk = reshape(k(:,:,pe1:pe2),2,[]).';
    wtmp = w(:,1:npe_per_slice);
    wtmp = wtmp(:);
    G = gpuNUFFT(tmpk.'*0.5/pi,wtmp,osf,wg,sw,[N,N],[],true);
    for c = 1:nchan
        tmpimg = G'*tmpdata(:,c);
        img(:,:,z,c) = reshape(tmpimg,N,N);
    end
end
m_gn = sqrt(sum(abs(img).^2,4));
toc;

%%
rawrite(m_gn, '../fig/img_gpunufft.ra');

%% Fessler's IRT
addpath('../contrib/irt');


%%
osf = 2; wg = 2; sw = 1;
imwidth = N;
img = zeros(N,N,nslices,nchan,'single');
tic;
for z = 1:nslices
    fprintf('irt z %d\n', z);
    pe1 = (z-1)*dpe + 1;
    pe2 = pe1 + npe_per_slice-1;
    tmpdata = double(reshape(data(:,pe1:pe2,:),[],nchan));
    %tmpdata(:,:) = 1.;
    tmpk = reshape(k(:,:,pe1:pe2),2,[]).';
    wtmp = w(:,1:npe_per_slice);
    wtmp = wtmp(:);
    %G = Gnufft({tmpk, [N N], wg*[2 2], osf*[N N], [N/2 N/2]});
    st = nufft_init(tmpk, [N N], wg*[2 2], osf*[N N], [N/2 N/2]);
    for c = 1:nchan
        tmpimg = nufft_adj(wtmp.*tmpdata(:,c),st);
        %tmpimg = G'*tmpdata(:,c);
        img(:,:,z,c) = reshape(tmpimg,N,N);
    end
end
m_irt = sqrt(sum(abs(img).^2,4));
toc;

%%
rawrite(m_gn, '../fig/img_irt.ra');

%% run this paper's code
!make
!./rr2d
!mv img_tron.ra ../fig/

%% load this paper's result
m_tron = single(abs(squeeze(raread('../fig/img_tron.ra'))));
m_irt = raread('../fig/img_irt.ra')/3;
m_gn = raread('../fig/img_gpunufft.ra')/1.5;

%%
zview = floor(nslices/3)+1;
yview = floor(N/2);
x_irt = normalize(fliplr(rot90(m_irt(:,:,zview))));
x_gn = normalize(fliplr(rot90(m_gn(:,:,zview))));
x_tron = normalize(fliplr(m_tron(:,:,zview)));
y_irt =normalize(fliplr(squeeze(m_irt(:,yview,:)).'));
y_gn = normalize(fliplr(squeeze(m_gn(:,yview,:)).'));
y_tron = normalize(fliplr(squeeze(m_tron(yview+2,:,:)).'));
fprintf('TODO: figure out why my code is shifted!\n');
figure(1)
x = [x_irt x_gn x_tron*1e10];
x = x(80:end-100,:);
imagesc(x);
imwrite(x,'../fig/axial.png');
colormap(gray);
axis image;
title('axial');

figure(2)
y = [y_irt, y_gn, y_tron];
y = y(10:end,:);
y = imresize(y, [950,size(y,2)]);
imshow(y,[]);
imwrite(y, '../fig/coronal.png');
colormap(gray);
title('coronal');
colorbar;

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
