%% Clear and close all figures
clear; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('data.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some datapoints
fprintf('First 10 examples from the dataset');
fprintf(' x = [%.0f %.0f], y = %.0f \n ', [X(1:10, :) y(1:10, :)]');

fprintf('Program paused. Press enter to continue \n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Feaures ...\n');

[X mu sigma] =  featureNormalize(X);

X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Alpha value
alpha = 0.01;
num_iters = 1000;

theta = zeros(3, 1);
[theta, J_History] = gradientDescent(X, y, theta, alpha, num_iters);

% Plotting the convergence
figure;
plot(1:numel(J_History), J_History, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);


% predicting the spending score of an 18 year old with an annual income of $41,000
new_values = [18, 41];
[new_values_normalized, new_values_mu, new_values_sigma] = featureNormalize(new_values);
spending_score = [1, new_values_normalized] * theta;

% test
fprintf(['Predicted spending score of an 18 year old, $41,000 annual income out of 100 ' ...
         '(using gradient descent):\n $%f\n'], spending_score);
         
         