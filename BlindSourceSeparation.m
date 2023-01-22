%% Tasks
%% task1
clear;
load("Music_Ozerov.mat");
%load("Shannon_Hurley.mat");
 
A= rand(3);
S=A*S;

 % Aest = AMUSE (S); % AMUSE algorithm run
SIR_Shannon_Hurley = 0;
 % SIR_Shannon_Hurley = CalcSIR (A, Aest ) %SIR calculation
 for i=1:10
     AestInv = AMUSEgit (S) ; % AMUSE algorithm from GIThub
     %AestInv = AMUSE (A,1)
    SIR_Shannon_Hurley = SIR_Shannon_Hurley + CalcSIR(A,inv(AestInv))
 end

SIR_Shannon_Hurley = SIR_Shannon_Hurley/10
 
Aest = inv(AestInv);
fs =15000;
Sest = inv(Aest)*S;
 
figure;
subplot(3,1,1)
plot(S(1,:))
title("Mixed signal 1");
 
subplot(3,1,2)
plot(S(2,:))
title("Mixed signal 2");
 
subplot(3,1,3)
plot(S(3,:))
title("Mixed signal 3");
 
 
 
figure;
subplot(3,1,1)
plot(Sest(1,:))
title("Separated bass");
% title("Separated piano");
subplot(3,1,2)
plot(Sest(2,:))
title("Separated classical guitar");
% title("Separated vocal");
 
subplot(3,1,3)
plot(Sest(3,:))
title("Separated electrical guitar");
%title("Separated drums");

player = audioplayer(Sest(1, :), fs);
% play(player);

soundsc ( Sest (1 ,:) , fs)
% soundsc ( Sest (2 ,:) ,fs)
% soundsc ( Sest (3 ,:) ,fs)

%% task2

addpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\pca_ica')); %Brian Moore mathworks

addpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\FastICA_25')); %Aalto university dept of Computer Science

rmpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\pca_ica')); %Brian Moore mathworks
rmpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\FastICA_25')); %Aalto university dept of Computer Science
%% Aalto
clear all;
rmpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\pca_ica')); %Brian Moore mathworks
rmpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\FastICA_25')); %Aalto university dept of Computer Science
addpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\FastICA_25')); %Aalto university dept of Computer Science

% load('Music_Ozerov.mat');
load('Shannon_Hurley.mat');
A = rand(3);
S = A*S;

SIR = 0;
tic
for i=1:10
    [Sica, Aica, W] = fastica(S);
    SIR = SIR + mean(CalcSIR(Aica, A));
end
toc = toc/10
SIR = SIR/10

%%
figure;
subplot 311;
plot(Sica(1,:));
title("Drums");
subplot 312;
plot(Sica(2,:));
title("Voice");
subplot 313;
plot(Sica(3,:));
title("Piano");
%%
figure;
subplot 311;
plot(S(1,:));
title("Mixed signal 1");
subplot 312;
plot(S(2,:));
title("Mixed signal 2");
subplot 313;
plot(S(3,:));
title("Mixed signal 3");


%% Brian Moore
clear all;
rmpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\pca_ica')); %Brian Moore mathworks
rmpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\FastICA_25')); %Aalto university dept of Computer Science
addpath(genpath('D:\onedrive\OneDrive\Dokumenty\polibudka\9\Machine Learning Methods\Lab\3\pca_ica')); %Brian Moore mathworks

load('Music_Ozerov.mat');
% load('Shannon_Hurley.mat');
A = rand(3);
S = A*S;

SIR = 0;
tic
for i=1:10
    [Sica, W, T, mu] = kICA(S, 3);
    SIR = SIR + mean(CalcSIR(A, (T\W')));
end
toc = toc/10
SIR = SIR/10


%% ICA - gradient descent
clear all;
close all;

load('Music_Ozerov.mat');
% load('Shannon_Hurley.mat');

SIR = 0;
tic
for i=1:10
    A = rand (3);
    S = A * S;


    xt = S - mean(S, 2); % centalization
    sizext = size(xt); % ( size of signal )
    Cx = xt * xt'/sizext(2); % correlation
    [Vx, Dx] = eigs (Cx, sizext(1)); % EVD
    z = inv(sqrt(Dx))*Vx' * xt; % Whitening

    % gradient
    grad = @(z, w)mean(z.*((w'*z).^3),2);

    % tolerance
    tol = 1e-6;
    % maximum number of allowed iterations
    maxiter = 100;
    % step size
    eta = 0.1;
    % optimization W matrix
    W = zeros(3);
    % iteration counter
    niter = 0;
    % perturbation
    delta = inf;

    for k = 1:3
        w = randn(3, 1);
        w = w/norm(w);
        % gradient descent algorithm :
        while and(niter <= maxiter, delta >= tol)
            % calculate gradient :
            g = grad(z, w);
            % next w
            wnew = w - eta*g;
            % orthogonalization
            gs = zeros(3, 1);
            for j = 1:k-1
                gs = gs + wnew'*W(:,j)*W(:,j);
            end
            wnew = wnew - gs;
            % normalization
            w = wnew/norm(wnew);
            % loop conditions
            niter = niter + 1;
            delta = norm(w-wnew);
        end
        W(:,k) = w;
    end

    Sica = W'* z;
    Aest = W'* Vx *inv(sqrt(Dx))*Vx';
    SIR = SIR + mean(CalcSIR(A, inv(Aest)));
end

toc = toc/10
SIR = SIR/10


%% ICA fixed point

clear all;
close all;

load('Music_Ozerov.mat');
% load('Shannon_Hurley.mat');

SIR = 0;
tic
for i=1:10
    A = rand(3);
    S = A * S;

    
    xt = S - mean(S, 2); % centalization
    sizext = size(xt); % ( size of signal )
    Cx = xt * xt'/sizext(2); % correlation
    [Vx , Dx] = eigs(Cx, sizext(1)); % EVD
    z = inv(sqrt(Dx))*Vx'*xt; % Whitening

    % maximum iterations
    maxiter = 100;

    W = zeros(3);

    for k = 1:3
        w = randn(3, 1);
        w = w/norm(w);
        for l = 1:maxiter
            % take step
            wnew = mean(z.*((w'*z).^3),2);
            % orthogonalization
            gs = zeros(3,1);
            for j = 1:k-1
                gs = gs + wnew'*W(:,j)*W(:,j);
            end
            wnew = wnew - gs;
            % normalization
            w = wnew/norm(wnew);
        end
        W(:,k) = w;
    end

    Sica = W'*z;
    Aest = W'*Vx*inv(sqrt(Dx))*Vx';
    SIR = SIR + mean(CalcSIR(A, inv(Aest)));
end

toc = toc/10
SIR = SIR/10



%% Algorithms

function [Sest, Aest] = AMUSE(S,tau);

    xt = S - mean(S ,2); % centalization
    sizext = size(xt); %(size of signal)
    Cx = xt*xt'/sizext(2); % correlation
    [Vx ,Dx] = eigs (Cx, sizext(1)); %EVD
    Z = inv(sqrt(Dx)) * Vx' * xt; % Whitening

    Z1 = Z(:, 1:sizext(2) - tau); %Z1
    Z2 = Z(:, 1 + tau:sizext(2)); %Z2 shifted by tau

    R = (Z1*Z2')/(sizext(2) - tau); % Autocorrelation
    R = (R+R')/2; % Symmetrization
    [V,D] = eigs(R, 3); %EVD

    Sest = V'*Z; %: final estimation
    Aest = S/Sest;
end

function [W] = AMUSEgit (X)
    % BSS using eigenvalue value decomposition
    % Program written by A. Cichocki and R. Szupiluk
    %
    % X [m x N] matrix of observed ( measured ) signals ,
    % W separating matrix ,
    % y estimated separated sources
    % p time delay used in computation of covariance matrices
    % optimal time - delay default p= 1
    %
    % First stage : Standard prewhitening

    [m,N]= size (X);
    if nargin ==1
        n=m;
    end

    Rxx =(X*X')/N;

    [Ux, Dx, Vx] = svd(Rxx);
    Dx = diag(Dx);
    % n=xxx;
    if n<m % under assumption of additive white noise and
        % when the number of sources are known or can a priori
        estimated
        Dx = Dx - real((mean(Dx(n+1:m))));
        Q = diag (real(sqrt(1./Dx(1:n))))*Ux(:,1:n)';
        %
    else % under assumption of no additive noise and when the
        % number of sources is unknown
        n = max(find(Dx>1e-199)); % Detection the number of sources
        Q = diag(real(sqrt(1./Dx(1:n))))*Ux(:,1:n)';
    end
    
    % else % assumes no noise
    % Q=inv( sqrtm (Rxx));
    % end;
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Second stage : Fast separation using sorting EVD
    % notation the same as used in the Chapter 4
    Xb = Q*X;
    p = 1;
    % paramter p can take here value different than 1
    % for example -1 or 2.
    N = max(size (Xb));
    Xb = Xb - kron(mean(Xb')',ones(1,N));

    Rxbxbp = (Xb(:,1:N-1)*Xb(:,2:N)')/(N-1);
    Rxbxbp = Rxbxbp +Rxbxbp';
    [Vxb Dxb ] = eig(Rxbxbp);
    [D1 perm ] = sort(diag(Dxb));
    D1 = flipud(D1);
    Vxb = Vxb(:,flipud(perm));
    W = Vxb'*Q;
    %y = Vxb ’ * x1;
end

function SIR = CalcSIR(A,Aest)
    
    % Sergio Cruces & Andrzej Cichocki 
    A = A*diag(1./(sqrt(sum(A.^2))+eps));
    Aest = Aest*diag(1./(sqrt(sum(Aest.^2)) + eps));

    %A=bsxfun(@rdivide,A,sum(A,1));
    %Aest=bsxfun(@rdivide,Aest,sum(Aest,1));

    for i = 1:size(Aest,2)    
        [MSE(i),ind] = min([sum(bsxfun(@minus,Aest(:,i),A).^2,1) sum(bsxfun(@plus,Aest(:,i),A).^2,1)]);
        %A(:,ind) = [];
    end
    SIR = -10*log10(MSE);
end

function kurt = my_kurtosis (S)
    x = S' - mean(S', 2);
    num = mean(x .^4);
    den = mean(x .^2).^2;
    kurt = num./den-3;
end


