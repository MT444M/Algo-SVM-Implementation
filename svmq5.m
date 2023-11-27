%%  ----------Classifieur polynomial pour le probleme des iris---------
clc;
clear;  
close all;

%% 
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
selected_classes = [2, 3];
idx_selected = ismember(c_encoded, selected_classes);
c_sous = c_encoded(idx_selected);
c_sous(c_sous == 2) = 1;
c_sous(c_sous == 3) = -1;
X_sous = X(idx_selected, 3:4);

X = X_sous;
y = c_sous;

%% Estimation des paramétres alpha
% -------Construction de la matrice de Gram avec le noyau polynomial
M = length(y);
H = zeros(M, M);
% Paramètre du noyau polynomial
beta = 3;
for i = 1:M
    for j = 1:M
        H(i, j) = y(i)*y(j)*(X(i, :)*X(j, :)' + 1)^beta;
    end
end

% Vecteur f
f = -ones(M, 1);

% Contraintes : alpha >= 0
lb = zeros(M, 1);

% Contraintes : y_m * alpha_m - epsilon_m >= 1
A = -eye(M);
b = -zeros(M, 1);
Aeq = zeros(M,M);
Aeq(1,:) = y';
beq = zeros(M,1); 

% Contraintes : epsilon_m >= 0
C = 1;
ub = C * ones(M, 1);

% Résolution du problème d'optimisation avec quadprog
options = optimset('Algorithm', 'interior-point-convex');
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

%% ---------------------------w et w0 ------------------------------------
% On ne peut pas remonter à w/ car phi est inconnu
% calcul de w0/
% Trouver un vecteur de support (peu importe lequel)
support_index = find(alpha > 0.1, 1);

% Calculer w0 avec noyau polynomial
w0 = 1/y(support_index) - ...
sum(alpha .* y .* (X(support_index, :) * X' + 1)'.^beta);


%% -------------------------Fonction de prédiction-------------------------
% Initialiser un vecteur pour stocker les prédictions
predictions = zeros(M, 1);

for i = 1:M
    % Calculer la partie dépendante des vecteurs de support avec noyau polynomial
    support_term = sum(alpha .* y.* (X(i, :) * X' + 1)'.^beta);
    
    % Calculer la fonction de décision avec noyau polynomial
    decision = support_term + w0;    
    % Prédiction basée sur la fonction de décision avec noyau polynomial
    predictions(i) = sign(decision);
end

% Visualisation des données et de la fonction de prédiction
figure()

% % Affichage )des points de données colorés par classe
% scatter(X(:, 1), X(:, 2), 50, y,'filled');
hold on;

% Affichage de la fonction de prédiction
scatter(X(predictions == 1, 1), X(predictions == 1, 2), 50, 'g'); % Points prédits comme classe 1% hold on;
scatter(X(predictions == -1, 1), X(predictions == -1, 2), 50, 'r'); % Points prédits comme classe -1

% Tracer les vecteurs de support
% Trouver tous les vecteurs de support
support_index = find(alpha > 0.1, 100);
scatter(X(support_index, 1), X(support_index, 2), 100, 'b', 'filled');

% Légende
legend('Classe 1', 'Classe -1', 'Vecteurs de support');
% Titre et étiquettes des axes
title('Prédiction de classe avec SVM à noyau gaussien');
xlabel('Caractéristique 1');
ylabel('Caractéristique 2');

hold off;

%%   --------------------Visualiser la décision------------------------

% Créer une grille d'échantillonnage pour les caractéristiques
x1_min = min(X(:, 1));
x1_max = max(X(:, 1));
x2_min = min(X(:, 2));
x2_max = max(X(:, 2));
[X1_grid, X2_grid] = meshgrid(linspace(x1_min, x1_max, M/2), linspace(x2_min, x2_max, M/2));

% Calculer les prédictions pour chaque point de la grille avec noyau polynomial
grid_predictions = zeros(size(X1_grid));
for i = 1:numel(X1_grid)
    support_term = sum(alpha .* y .* (X * [X1_grid(i), X2_grid(i)]' + 1).^beta);
    decision = support_term + w0;
    grid_predictions(i) = sign(decision);
end

% Visualisation des données et de la fonction de prédiction
figure;

% Affichage de la fonction de prédiction avec colormap
imagesc(linspace(x1_min, x1_max, 100), linspace(x2_min, x2_max, 100), reshape(grid_predictions, size(X1_grid)));
colormap('summer');
colorbar;

hold on;
% Affichage des points de données colorés par classe
scatter(X(1:M/2,1),X(1:M/2,2),'b') 
scatter(X(M/2+1:end,1),X(M/2+1:end,2),'r') 
% Tracer les vecteurs de support
%scatter(X(support_index, 1), X(support_index, 2), 50, 'm', 'filled');

% Légende
legend('Classe 1', 'Classe -1', 'Vecteurs de support');

% Titre et étiquettes des axes
title('Prédiction de classe avec SVM à noyau gaussien');
xlabel('Caractéristique 1');
ylabel('Caractéristique 2');

hold off;

%% ---------------------------Évolution des performances----------------------------

% Calcul des prédictions pour l'ensemble de données


% Calcul de la précision
accuracy = sum(predictions == y) / M;

% Calcul de la matrice de confusion
confusion_matrix = confusionmat(y, predictions);

% Calcul de la précision (precision)
precision = confusion_matrix(1, 1) / (confusion_matrix(1, 1) + confusion_matrix(2, 1));

fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);

% Affichage de la matrice de confusion
disp('Matrice de Confusion :');
disp(confusion_matrix);

