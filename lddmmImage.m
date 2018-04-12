% This code written by Daniel Tward, Johns Hopkins University, 2016
% deform image I to match J
% regularization energy is \|Lv\|^2_{L_2}/2, L = Id - alpha^2 Laplacian,
% matching energy is \|I(phiinv) - J\|^2_{L_2}/2/sigma^2
% flow is discretized to nT timesteps
% optimization using gradient descent with stepsize epsilon for nIter steps
% energy gradient is -(I - J) grad(I) det(D phi1t )
function [It,Jt,phiInvtx,phiInvty,phiInv1tx,phiInv1ty,vtx,vty] = lddmmImage(I,J,alpha,sigma,nT,epsilon,nIter)
% initialize velocity field
vtx = zeros([size(I) nT]);
vty = zeros([size(I) nT]);
for i = 1 : nIter
    % flow diffeomorphisms
    [phiInvtx,phiInvty] = calculatePhiInvtFromV(vtx,vty);
    [phiInv1tx,phiInv1ty] = calculatePhiInv1tFromV(vtx,vty);
    % deform images
    It = applyPhitToImage(I,phiInvtx(:,:,2:end),phiInvty(:,:,2:end));
    Jt = applyPhitToImage(J,phiInv1tx(:,:,2:end),phiInv1ty(:,:,2:end));
    % energy (cost function)
    E = sum(sum(sum(applyPowerOfA(vtx,alpha,1).*vtx + applyPowerOfA(vty,alpha,1).*vty)))/nT/2 + sum(sum((It(:,:,end)-J).^2))/2/sigma^2;
    fprintf('Iter %d of %d, energy is %g\n',i,nIter,E);
    % determinant of Jacobian
    detJac = calculateDeterminantOfJacobian(phiInv1tx(:,:,2:end),phiInv1ty(:,:,2:end));
    % gradient of It
    [gradItx,gradIty] = calculateImageGradient(It);
    % gradient of matching term
    gradMatchx = applyPowerOfA( -(It - Jt).*detJac.*gradItx/sigma^2, alpha, -1);
    gradMatchy = applyPowerOfA( -(It - Jt).*detJac.*gradIty/sigma^2, alpha, -1);
    % energy gradient
    gradx = vtx + gradMatchx;
    grady = vty + gradMatchy;
    % update velocity
    vtx = vtx - gradx*epsilon;
    vty = vty - grady*epsilon;    
end

function [phiInvtx,phiInvty] = calculatePhiInvtFromV(vtx,vty)
nT = size(vtx,3);
[X,Y] = meshgrid(1:size(vtx,2),1:size(vtx,1));
phiInvtx = repmat(X,[1 1 nT+1]);
phiInvty = repmat(Y,[1 1 nT+1]);
for i = 1 : nT
    % make temporaries for logical indexing
    vx = vtx(:,:,i);
    vy = vty(:,:,i);
    phiInvx = phiInvtx(:,:,i);
    phiInvy = phiInvty(:,:,i);
    % update by estimating where particle was at last timestep
    phiInvx_ = interp2(phiInvx,X-vx/nT,Y-vy/nT,'linear',NaN);
    phiInvy_ = interp2(phiInvy,X-vx/nT,Y-vy/nT,'linear',NaN);
    % identity boundary conditions
    ind = isnan(phiInvx_) | isnan(phiInvy_);
    phiInvx_(ind) = phiInvx(ind)-vx(ind)/nT;
    phiInvy_(ind) = phiInvy(ind)-vy(ind)/nT;
    phiInvtx(:,:,i+1) = phiInvx_;
    phiInvty(:,:,i+1) = phiInvy_;
end

function [phiInv1tx,phiInv1ty] = calculatePhiInv1tFromV(vtx,vty)
% follow the negative of the velocity backwards in time
[phiInv1tx,phiInv1ty] = calculatePhiInvtFromV(flip(-vtx,3),flip(-vty,3));
phiInv1tx = flip(phiInv1tx,3);
phiInv1ty = flip(phiInv1ty,3);

function It = applyPhitToImage(I,phiInvtx,phiInvty)
% deform image by composing with inverse (interpolating at specified points)
It = zeros(size(phiInvtx));
for i = 1 : size(phiInvtx,3)
    It(:,:,i) = interp2(I,phiInvtx(:,:,i),phiInvty(:,:,i),'linear',0);
end

function p = applyPowerOfA(v,alpha,power)
% A = (1 - alpha^2 Laplacian)^2, use discrete Laplacian in Fourier domain
[FX,FY] = meshgrid((0:size(v,2)-1)/size(v,2),(0:size(v,1)-1)/size(v,1));
Apow = ( 1 - alpha^2*2*(cos(2*pi*FX) + cos(2*pi*FY) - 2 ) ).^(2*power);
p = ifft2( bsxfun(@times, fft2(v), Apow), 'symmetric');

function [gradItx,gradIty] = calculateImageGradient(It)
gradItx = zeros(size(It));
gradIty = zeros(size(It));
for i = 1 : size(It,3)
    [gradItx(:,:,i),gradIty(:,:,i)] = gradient(It(:,:,i));
end

function detJac = calculateDeterminantOfJacobian(phiInv1tx,phiInv1ty)
[phiInv1tx_x,phiInv1tx_y] = calculateImageGradient(phiInv1tx);
[phiInv1ty_x,phiInv1ty_y] = calculateImageGradient(phiInv1ty);
detJac = phiInv1tx_x.*phiInv1ty_y - phiInv1tx_y.*phiInv1ty_x;
