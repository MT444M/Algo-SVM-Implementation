%%  ------------------ non lineaire------------------------
clc;
clear;
close all;

%% -------------------------création des données---------------------------
rng(1); % For reproducibility
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

%Generate 100 points uniformly distributed in the annulus. The radius is again proportional to a square root, this time a square root of the uniform distribution from 1 through 4.
r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

%Plot the points, and plot circles of radii 1 and 2 for comparison.
figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

X = [data1;data2];
c = ones(200,1);
c(1:100) = -1;

%% Estimation des paramétres alpha
% -------Construction de la matrice de Gram avec le noyau gaussien
M = length(c);
H = zeros(M, M);
sigma = 1;

for i = 1:M
    for j = 1:M
        H(i, j) = c(i) * c(j) * exp(-sum((X(i, :) - X(j, :)).^2) / (2 * sigma^2));
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
Aeq(1,:) = c';
beq = zeros(M,1); 

% Contraintes : epsilon_m >= 0
C = inf;
ub = C * ones(M, 1);

% Résolution du problème d'optimisation avec quadprog
options = optimset('Algorithm', 'interior-point-convex');
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

%% ---------------------------w et w0 ------------------------------------
% On ne peut pas remonter à w/ car phi est inconnu
% calcul de w0/
% Trouver un vecteur de support (peu importe lequel)
support_index = find(alpha > 0.1, 1);

% Calculer w0
w0 = 1/c(support_index) - sum(alpha .* c .*  exp(-pdist2(X, X(support_index, :), 'euclidean').^2 / (2 * sigma^2)));

%% -------------------------Fonction de prédiction-------------------------
% Initialiser un vecteur pour stocker les prédictions
predictions = zeros(M, 1);

for i = 1:M
    % Calculer la partie dépendante des vecteurs de support
    support_term = sum(alpha .* c .* exp(-pdist2(X, X(i, :), 'euclidean').^2 / (2 * sigma^2)));
    % Calculer la fonction de décision
    decision = support_term + w0;
    
    % Prédiction basée sur la fonction de décision
    predictions(i) = sign(decision);
end

% Visualisation des données et de la fonction de prédiction
figure;

% Affichage des points de données colorés par classe
%scatter(X(:, 1), X(:, 2), 50, c,'' 'filled');
hold on;

% Affichage de la fonction de prédiction
scatter(X(predictions == 1, 1), X(predictions == 1, 2), 50, 'g'); % Points prédits comme classe 1
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
[X1_grid, X2_grid] = meshgrid(linspace(x1_min, x1_max, 100), linspace(x2_min, x2_max, 100));

% Calculer les prédictions pour chaque point de la grille
grid_predictions = zeros(size(X1_grid));
for i = 1:numel(X1_grid)
    support_term = sum(alpha .* c .*exp(-pdist2(X, [X1_grid(i), X2_grid(i)], 'euclidean').^2 / (2 * sigma^2)) );
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
scatter(X(support_index, 1), X(support_index, 2), 50, 'm', 'filled');

% Légende
legend('Classe 1', 'Classe -1', 'Vecteurs de support');

% Titre et étiquettes des axes
title('Prédiction de classe avec SVM à noyau gaussien');
xlabel('Caractéristique 1');
ylabel('Caractéristique 2');

hold off;

%%


% Calcul de la précision
accuracy = sum(predictions == c) / M;

% Calcul de la matrice de confusion
confusion_matrix = confusionmat(c, predictions);

% Calcul de la précision (precision)
precision = confusion_matrix(1, 1) / (confusion_matrix(1, 1) + confusion_matrix(2, 1));

fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);

% Affichage de la matrice de confusion
disp('Matrice de Confusion :');
disp(confusion_matrix);



