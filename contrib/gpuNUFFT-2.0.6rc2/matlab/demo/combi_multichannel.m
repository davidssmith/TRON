%% Reconstruction of 3D radial vibe data
clear all; close all; clc; clear classes;

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));

%% Reconstruction parameters
maxitCG = 10;
alpha = 1e-6;
tol = 1e-6;
display = 1;

%% Load data
load /home/dss/Data/combi/CMT_XTEND_20141216/1231_QBC_GA_12mm_15_3.7_min.mat;
[nCh,nFE,nPE]=size(data);
rawdata = shiftdim(data,1);
%rawdata = reshape(rawdata,[nPE*nFE,nCh]);

%% Data parameters
N=512;
npe_per_slice = nFE/2;
dpe = 128;

useGPU = true;
useMultiCoil = 1;
nSl = floor((nPE - npe_per_slice) / dpe);
disp_slice=nSl/2;

%%
% rawdata is 207x3182x8
% k is 658674x3
% w is 3182 x 207

%% k-space coordinates: golden angle radial
k = zeros(3,nFE,nPE);
r = [0:nFE-1]/nFE - 0.5;
for pe = 1:nPE
    theta = 111.246*(pe-1)*pi / 180;
    k(1,:,pe) = r *cos(theta);
    k(2,:,pe) = r* sin(theta);
end
%k = reshape(k,3,[]);

%% assume Ram-Lak SDC for now
w = zeros(nFE,nPE);
for pe = 1:nPE
    for ro = 1:nFE
        w(ro,pe) = 2*(abs(ro - nFE/2) + 1) / (nFE+1);
    end
end
w = w.';

%% Regridding operator GPU without coil sensitivities for now
osf = 1; wg = 3; sw = 2;
imwidth = N;

for z = 1:1
    disp('Generate NUFFT Operator without coil sensitivities');
    pe1 = (z-1)*dpe+1;
    pe2 = pe1 + npe_per_slice;
    tmpdata = reshape(rawdata(:,pe1:pe2,:),[],nCh);
    tmpk = reshape(k(:,:,pe1:pe2),3,[]);
    tmpw = w(pe1:pe2,:).';
    FT = gpuNUFFT(tmpk,tmpw(:),osf,wg,sw,[N,N,nSl],[],true);
    
    for ii=1:nCh
        img_sens(:,:,:,ii) = FT'*tmpdata(:,ii);
    end
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
