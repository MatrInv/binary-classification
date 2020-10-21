
close all

% Données d'apprentissage
m1 = 50;
m = 2*m1;
X1 = 3 + randn(m1,2);
X2 = -3 + randn(m1,2);
X = [X1;X2];
Y = ones(m,1);
Y(m1+1:end) = -1;

% Grille de points de test pour visualiser la frontière
[Xt1, Xt2] = meshgrid(-6:0.3:6,-6:0.3:6);
Ntest = size(Xt1,1) * size(Xt1,2);
Xtest = [reshape(Xt1, Ntest,1), reshape(Xt2,Ntest,1)];

% Apprentissage SVM
C = 100;
model = fitcsvm( X, Y, 'BoxConstraint', C );

% Classification de la grille de points de test

Ypred = model.predict(Xtest);
%Ypred = zeros(size(Xtest,1),1);	% A modifier !

% Affichage : 
figure;
hold on;
gscatter(X(:,1), X(:,2), Y);
gscatter(Xtest(:,1), Xtest(:,2), Ypred );

% Calcul de w et affichage de la droite de séparation
a = model.Alpha;
x = model.SupportVectors;
y = Y(model.IsSupportVector);
w = x'*(a.*y);

b = model.Bias;

x1 = -5:5;
droite = (-w(1)*x1-b)/w(2);
plot(x1,droite,"b");
hold on;

% calcul de la marge
delta = 1/norm(w)
margeSup = (-w(1)*x1-b+1)/w(2);
margeInf = (-w(1)*x1-b-1)/w(2);

% Tracer les bords de la marge 
plot(x1,margeSup,"b--");
hold on;
plot(x1,margeInf,"b--");
hold on;

