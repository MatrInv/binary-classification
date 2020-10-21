close all
clear all
load defautsrails.mat

% Boucle pour apprendre les 4 classifieurs SVM binaires 
%	et calculer leur erreur d'apprentissage
%	puis calculer l'erreur d'apprentissage du classifieur multi-classe
C = 1;
scores=[];
model={};
for k=1:4
    Yk = 2*(Y==k)-ones(size(Y));
    model{k} = fitcsvm( X, Yk, 'BoxConstraint', C );
    [Ykpred,scorek] = model{k}.predict(X);

    %taux d'erreur de chaque classifieur binaire
    mean(Yk ~= Ykpred);
    scores = [scores, scorek(:,2)];
    
    mean(Yk ~= Ykpred)

end

[M,Ypred] = max(scores,[],2);

%taux d'erreur du classifieur multi-classe
mean(Y ~= Ypred)


