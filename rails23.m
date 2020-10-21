close all
clear all
load defautsrails.mat

% Boucle pour apprendre les 4 classifieurs SVM binaires 
%	et calculer leur erreur d'apprentissage
%	puis calculer l'erreur d'apprentissage du classifieur multi-classe
tic

C = 10;
Ypred = [];
error = [0,0,0,0];
for i=1:140

    scores=[];
    model={};
    
    Yi = Y(i,:);
    Ynoi = Y;
    Ynoi(i,:) = [];
    
    Xi = X(i,:);
    Xnoi = X;
    Xnoi(i,:) = [];
    
    for k=1:4
        Yk = 2*(Ynoi==k)-ones(size(Ynoi));
        model{k} = fitcsvm( Xnoi, Yk, 'BoxConstraint', C );
        [Ykpred,scorek] = model{k}.predict(Xi);
        scores = [scores, scorek(:,2)];
        if (Ykpred == 1 && Yi ~= k) ||(Ykpred == -1 && Yi == k)
            error(k) = error(k) + 1;
        end
    end

    [M,Yipred] = max(scores,[],2);
    Ypred = [Ypred ; Yipred];  
    
end

%taux d'erreur des classifieurs binaire
classifier_k_error = error / 140

%taux d'erreur du classifieur multi-classe
multiclass_error = mean(Y ~= Ypred)

toc




% Test LOO : boucle sur tous les exemples


