function [U, Vs, alpha, objHistory] = EMKCF(Ks, nCluster)

nKernel = length(Ks);
nSmp = size(Ks{1}, 1);


%*******************************************
% Init alpha
%*******************************************
aa = zeros(nKernel, 1);
tmp = Ks{1};
aa(1) = sum(diag(tmp));
for iKernel = 2:nKernel
    aa(iKernel) = sum(diag(Ks{iKernel}));
    tmp = tmp + Ks{iKernel};
end
tmp = (tmp + tmp')/2;
tmp = tmp/nKernel;
if nSmp < 200
    opt.disp = 0;
    [U, ~] = eigs(tmp, nCluster, 'la', opt);
else
    tmp = tmp - spdiags(diag(tmp), 0, nSmp, nSmp);
    U = spectral_cluster_pm_k(tmp, nCluster);
end
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1, nCluster);
[label, ~, ~] =  kmeanspp(U_normalized', nCluster);
U = ind2vec(label)' + 0.2;
clear tmp;

es = ones(1, nKernel);
alpha = sqrt(es)/sum(sqrt(es));

converges = false;
iter = 0;
maxIter = 100;
objHistory = [];
while ~converges
    iter = iter + 1;
    
    %*******************************************
    % Optimize Vs
    %*******************************************
    Vs = cell(1, nKernel);
    KVs = cell(1, nKernel);
    for iKernel = 1:nKernel
        KYi = Ks{iKernel}' * U; % m * c
        [U2, ~, V2] = svd(KYi, 'econ');
        Vs{iKernel} = U2 * V2'; % m * c
        KVs{iKernel} = Ks{iKernel} * Vs{iKernel}; % n * c
    end
    %     obj = compute_obj(Ks, Y, Vs, alpha);
    %     objHistory = [objHistory; obj]; %#ok
    
    
    %*******************************************
    % Optimize Y
    %*******************************************
    B = zeros(nSmp, nCluster);
    for iKernel = 1:nKernel
        B = B + (1/alpha(iKernel)) * KVs{iKernel};
    end
    B = -B;
    A = 1/alpha(1) * Ks{1};
    for iKernel = 2:nKernel
        A = A + (1/alpha(iKernel)) * Ks{iKernel};
    end
    A_pos = max(A, 0);
    A_neg = (abs(A) - A)/2;
    for iter2 = 1:10
        A_posU = A_pos * U;
        A_negU = A_neg * U;
        tmp1 = (sqrt(B.^2 + 4 * (A_posU .* A_negU)) - B);
        U = U .* tmp1 ./ max(2 * A_posU, eps);
    end
    %     obj = compute_obj(Ks, Y, Vs, alpha);
    %     objHistory = [objHistory; obj]; %#ok
    
    
    %*******************************************
    % Optimize alpha
    %*******************************************
    ab = zeros(nKernel, 1);
    bb = zeros(nKernel, 1);
    for iKernel = 1:nKernel
        ab(iKernel) = sum(sum( U .* KVs{iKernel}));
        KU = Ks{iKernel} * U;
        bb(iKernel) = sum(sum( U .* KU));
    end
    es = aa - 2*ab + bb;
    alpha = sqrt(es)/sum(sqrt(es));
    %*******************************************
    % Compute obj
    %*******************************************
    obj = sum(sum((1./alpha) .* es));
    objHistory = [objHistory; obj]; %#ok
    
    
    if (iter>20) && (  abs( objHistory(iter-1) - objHistory(iter) )  /abs(objHistory(iter-1) ) <1e-5 || iter>maxIter)
        converges = true;
    end
    
end

end
