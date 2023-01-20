%% Code



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


