function idxCV = CVsplit( N, K )
%CVsplit creates a matrix of N randomized indices with K columns.

    idx = randperm(N);
    Nk = floor(N/K);
    idxCV = zeros(Nk,K);
    for k = 1:K
        idxCV(:,k) = idx(1+(k-1)*Nk:k*Nk);
    end

end

