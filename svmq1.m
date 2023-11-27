%%  -------------------SVM lineaire et non lineaire------------------------
clc;
clear;
close all;

%% ------------------CAS LINEAIREMENT SEPARABLE, PRIMAL------------------

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

%% --------------------------PROBLEME PRIMAL-------------------------------
% Identification des paramètres du modèle avec ma fonction quadprog de
% Mat lab
M = size(X_sous, 1);
N = size(X_sous, 2);

% Construction de la matrice H
H = eye(N + 1); 
H(1,1) = 0; 

 % Construction de la matrice f
f = zeros(N + 1, 1);

 % Construction de la matrice A et du vecteur b
A = diag(c_sous) * [ones(M, 1), X_sous];
b = -ones(M, 1);

% Résolution du problème d'optimisation avec quadprog
options = optimset('Algorithm', 'interior-point-convex');
theta =  quadprog(H, f, A, b, [], [], [], [], [], options);

% Extraire w et w0 de x
w0 = theta(1);
w = theta(2:end);

%% 4 ------Affichage de la droite de séparation----------

% Déterminer la droite de séparation
x_values = 0:0.1:2;
y_values = -(theta(1) + theta(2) * x_values) / theta(3);

figure()
% Tracer les points de données
scatter(X_sous(:, 2), X_sous(:, 1), 50, c_sous,'filled');

hold on;
% Tracer la droite de séparation
plot(x_values,y_values, 'r');
xlabel('Longueur pétale');
ylabel('Largeur pétale');
title('Droite de séparation ');
legend('données','Droite de Séparation');


%% ------------------------FONCTION DE DECISION---------------------------

% Calcul de la fonction de décision
 decision_values = -(X_sous * w + w0);

 f_decision = c_sous.*decision_values;

% Identifie les vecteurs de support
vect_support = find(abs(1 - f_decision) < 1e-6); % 1e-6 ~ 0

%----------------------- Tracer les points de données----------------------
figure();
scatter(X_sous(:, 2), X_sous(:, 1),50, c_sous, 'filled');
% Marquer les vecteurs de support en carrés rouges
hold on;
scatter(X_sous(vect_support(:), 2), X_sous(vect_support(:), 1), 's', 'red');
% Tracer la droite de séparation
plot(x_values, y_values, 'r');
xlabel('Longueur pétale');
ylabel('Largeur pétale');
title('Droite de séparation avec vecteurs de support');
legend('données', 'Vecteurs de Support', 'Droite de Séparation');

%%  ---------------------------Image COULEUR-------------------------------
%création de la grille
x1min=min(X_sous(:,1));
x1max=max(X_sous(:,1));
x2min=min(X_sous(:,2));
x2max=max(X_sous(:,2));
x1=x1min:0.01:x1max;
x2=x2min:0.01:x2max;
[Xg,Yg] = meshgrid(x1,x2);
f=theta(2)*Xg+theta(3)*Yg+theta(1);
fp=-ones(size(Xg));
fp(f>=0)=1;

% affichage 
figure()
hold on;
imagesc(x1,x2,fp);
axis xy
colormap('summer')
colorbar
%les données
scatter(X_sous(:, 1), X_sous(:, 2), 'filled');
% Marquer les vecteurs de support en carrés rouges
scatter(X_sous(vect_support(:), 1), X_sous(vect_support(:), 2), 's', 'red');
xlabel('Longueur pétale');
ylabel('Largeur pétale');
title('hyperplan de séparation');
legend('données', 'Vecteurs de Support');

