%% Reconstruction of 3D radial vibe data
clear all; close all; clc; clear classes;

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));

%% Data parameters
N=512;
nSl=83;
nFE=512;
nCh=6;
disp_slice=nSl/2;
useGPU = true;
useMultiCoil = 1;

%% Reconstruction parameters
maxitCG = 10;
alpha = 1e-6;
tol = 1e-6;
display = 1;

%% Load data
load /home/dss/Data/combi/CMT_XTEND_20141216/1231_QBC_GA_12mm_15_3.7_min.mat;
[nCh,nFE,nPE]=size(data);
rawdata = shiftdim(data,1);
rawdata = reshape(rawdata,[nPE*nFE,nCh]);

%%
k = zeros(3,nFE,nPE);

for pe = 1:nPE
    theta = 111.246*(pe-1)*pi / 180;
    for ro = 1:nFE
        k(1:2,ro,pe) = (ro/nFE - 0.5) *[cos(theta), sin(theta)];
    end
end

%% Regridding operator GPU without coil sensitivities for now
disp('Generate NUFFT Operator without coil sensitivities');
osf = 2; wg = 3; sw = 8;
imwidth = N;
FT = gpuNUFFT(k',col(w(:,:,1)),osf,wg,sw,[N,N,nSl],[],true);

for ii=1:nCh
    img_sens(:,:,:,ii) = FT'*rawdata(:,ii);
end

%% Estimate sensitivities
disp('Estimate coil sensitivities.');
% Terribly crude, but fast
img_sens_sos = sqrt(sum(abs(img_sens).^2,4));
senseEst = img_sens./repmat(img_sens_sos,[1,1,1,nCh]);

% Use this instead for more reasonable sensitivitities, but takes some time
% for ii=1:nSl
%     disp(['Slice ', num2str(ii), '/', num2str(nSl)]);
%     [~,senseEst(:,:,ii,:)]=adapt_array_2d(squeeze(img_sens(:,:,ii,:)));
% end

%% Redefine regridding operator GPU including coil sensitivities
disp('Generate NUFFT Operator with coil sensitivities');
FT = gpuNUFFT(k',col(w(:,:,1)),osf,wg,sw,[N,N,nSl],senseEst,true);

%% Forward and adjoint transform
tic
img_comb = FT'*rawdata;
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);
% figure,imshow(abs(img_comb(:,:,disp_slice)),[]); title('Regridding');
% figure,kshow(abs(fft2c(img_comb(:,:,disp_slice)))); title('Regridding k-space');

tic
test = FT*img_comb;
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);

%% CGSENSE Reconstruction
mask = 1;
tic
img_cgsense = cg_sense_3d(rawdata,FT,senseEst,mask,alpha,tol,maxitCG,display,disp_slice,useMultiCoil);
timeCG = toc;
disp(['Time CG SENSE: ', num2str(timeCG), ' s']);

%% Display
figure;
subplot(1,2,1); imshow(abs(img_comb(:,:,disp_slice)),[]); title('Regridding');
subplot(1,2,2); imshow(abs(img_cgsense(:,:,disp_slice)),[]); title('CGSENSE');
