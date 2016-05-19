function [ XTr, yTr, XVal, yVal ] = getTrVal( X, y, k, idxCV )
%getTrVal gets the validation set k given a CVidx matrix 

    idxTe = idxCV(:,k);
    idxTr = idxCV(:,[1:k-1 k+1:end]);
    idxTr = idxTr(:);
    yVal = y(idxTe);
    XVal = X(idxTe,:);
    yTr = y(idxTr);
    XTr = X(idxTr,:);


end

