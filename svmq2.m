%%  -------------------SVM lineaire et non lineaire------------------------
clc;
clear;
close all;

%% ------------------CAS LINEAIREMENT SEPARABLE, DUAL------------------
X = meas;
c = species;

% 1--- Encoder les classes en 1, 2 et 3
c_encoded = zeros(length(c), 1);
classes = unique(c);
for i = 1:length(classes)
    c_encoded(ismember(c, classes{i})) = i;
end

%3 classif binaire:
% Sélectionner uniquement les classes 1 et 2
selected_classes = [1, 2];
idx_selected = ismember(c_encoded, selected_classes);
c_sous = c_encoded(idx_selected);
c_sous(c_sous == 2) = -1;
X_sous = X(idx_selected, 3:4);

X = X_sous;
y = c_sous;
%%  PROBLEME DUAL

M = size(X, 1);
    
% Construction de la matrice de Gram Q
H = (y * y') .* (X * X');

% Vecteur f
f = -ones(M, 1);

% Contraintes : alpha >= 0 et sum(alpha .* y) = 0
A = -eye(M);
b = zeros(M, 1);
Aeq = y';
beq = 0;

% Résolution du problème d'optimisation avec quadprog
options = optimset('Algorithm', 'interior-point-convex');
alpha = quadprog(H, f, A, b, Aeq, beq, [], [], [], options);

%%  --------w et w_0-----------------
% Calcul de w et w0 à partir des alphas
w = sum((alpha .* y) .* X)';
w_0 = (1/y(45)) - w' * X(45,:)';


% Déterminer la droite de séparation
x_values = 0:0.1:5;
y_values = -(w_0 + w(1) * x_values) / w(2);

figure()
% Tracer les points de données
scatter(X(:, 1), X(:, 2),50,c_sous, 'filled');
hold on;
% Tracer la droite de séparation
plot(x_values,y_values, 'r');
xlabel('Longueur pétale');
ylabel('Largeur pétale');
title('Droite de séparation ');
legend('données','Droite de Séparation');

%% 5 ------------------------- FONCTION DE DECISION------------------------

X_prime = ones(M,3);
X_prime(:,1:2) = X;

f = X_prime * [w ;w_0];
fd_1 = f(1:50);
fd_2 = f(51:end);

vec_supp1 = find(fd_1 == min(fd_1));
vec_supp2 = 50 + find(fd_2 == max(fd_2)) ;

%----------------------- Tracer les points de données----------------------
figure();
scatter(X_sous(:, 1), X_sous(:, 2),50, c_sous, 'filled');
% Marquer les vecteurs de support en carrés rouges
hold on;
scatter(X_sous(vec_supp1, 1), X_sous(vec_supp1, 2), 'sr');
scatter(X_sous(vec_supp2, 1), X_sous(vec_supp2, 2), 'sr');

% Tracer la droite de séparation
plot(x_values, y_values, 'r');
xlabel('Longueur pétale');
ylabel('Largeur pétale');
title('Droite de séparation avec vecteurs de support');
legend('données', 'Vecteurs de Support', 'Droite de Séparation');

%% -------------------------IMMAGE BINAIRE--------------------------------
%Script pris de imagefbinaire.m
x1min=min(X(:,1));
x1max=max(X(:,1));
x2min=min(X(:,2));
x2max=max(X(:,2));
x1=x1min:0.01:x1max;
x2=x2min:0.01:x2max;
[Xg,Yg] = meshgrid(x1,x2);
f=w(1)*Xg+w(2)*Yg+w_0;
fp=-ones(size(Xg));
fp(f>=0)=1;

y_values = -(w_0 + w(1) * x1) / w(2);
figure ()
imagesc(x1,x2,fp);
axis xy
colormap('summer')
colorbar

hold on 
plot(X(1:50,1),X(1:50,2),'ob') 
plot(X(51:end,1),X(51:end,2),'om') 
scatter(X(vec_supp1,1),X(vec_supp1,2),'sr','filled') 
scatter(X(vec_supp2,1),X(vec_supp2,2),'sc','filled') 
plot(x1, y_values, 'r');
xlabel('Longueur pétale');
ylabel('Largeur pétale');
title('hyperplan de séparation: problème dual');
legend('classe1','classe -1', 'Vecteurs de Support');

