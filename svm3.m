
close all

% Données d'apprentissage
m1 = 50;
m = 2*m1;
X1 = 3 + 4*randn(m1,2);
X2 = -3 + randn(m1,2);
X = [X1;X2];
Y = ones(m,1);
Y(m1+1:end) = -1;

% Grille de points de test pour visualiser la frontière
[Xt1, Xt2] = meshgrid(-6:0.3:6,-6:0.3:6);
Ntest = size(Xt1,1) * size(Xt1,2);
Xtest = [reshape(Xt1, Ntest,1), reshape(Xt2,Ntest,1)];

% Apprentissage SVM
kernel = 'rbf';
sigma = 8;
C = 1;

svm = fitcsvm(X,Y,'KernelFunction', kernel, 'KernelScale',sigma, 'BoxConstraint', C);

% Classification de la grille de points de test

Ypred =  svm.predict(Xtest);

% Affichage : 
figure;
hold on;
gscatter(X(:,1), X(:,2), Y);
gscatter(Xtest(:,1), Xtest(:,2), Ypred );

