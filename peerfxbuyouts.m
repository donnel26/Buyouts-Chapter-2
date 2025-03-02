% Clear workspace and command window
clear; clc;

% Set number of households
N = 4;  % Total number of households in the model

% Household characteristics (X) - income levels
X = [0.5; 0.7; 0.6; 0.8];  % A column vector representing the characteristics of each household

% Peer influence matrix (W)
% This matrix represents the influence each household has on its peers.
W = [0 0.5 0.5 0;   % Household 1 influences Households 2 and 3
     0.5 0 0.5 0;   % Household 2 influences Households 1 and 3
     0.5 0.5 0 0;   % Household 3 influences Households 1 and 2
     0 0 0.5 0];    % Household 4 influences Household 3

% Contextual coefficients (alpha, delta, gamma)
alpha = 0.4;  % Coefficient for household characteristics; how i's characteristics drive decision
delta = 0.3;  % Coefficient for peer influence characteristics; contextual/exogenous effects coefficient; how i's peers' characteristics drive decision
gamma = 0.2;  % Coefficient for the household's own expectation; endogenous effects coefficient; how i's peers' decisions drive decision

% Initial guess for expectations (M)
M = [0.5; 0.6; 0.7; 0.8];  % Initial expectations of each household regarding their behavior

% Iteration settings
max_iter = 100;       % Maximum number of iterations for convergence
tolerance = 1e-6;     % Tolerance level for convergence check

% Iterate to update M until convergence
for iter = 1:max_iter
    M_old = M; % The current values of M are stored in M_old to compare later and check for convergence.
    for i = 1:N % Iterate through each household
        % Calculate the expected outcome for household i, does this for
        % i=1,..., N.
        new_Mi = X(i) * alpha + W(i,:) * X * delta + gamma * M(i);
        
        % Apply the tanh function to compute the new expectation for household i
        M(i) = tanh(new_Mi);
    end
    
    % Check for convergence: if the maximum change in M is less than the tolerance, stop iterating
    if max(abs(M - M_old)) < tolerance
        break;
    end
end

% Display the final expectations
disp('Final expectations (M):');
disp(M);

% Binary outcome variable
z = [1; -1; 1; -1]; % 1: participate in buyout, -1: don't participate in buyout

% Define the log-likelihood function for MLE
log_likelihood = @(params) -sum(log(1 ./ (1 + exp(-z .* (X * params(1) + W * X * params(2) + params(3) * M)))));

% Optimize parameters using MLE
params_init = [alpha; delta; gamma]; % Initial parameter values
options = optimset('GradObj', 'off', 'MaxIter', 400);
[params_hat, logL] = fminunc(log_likelihood, params_init, options);

% Display results
disp('Estimated parameters (alpha, delta, gamma):');
disp(params_hat);
disp('Final log-likelihood value:');
disp(-logL);
%% 
[params_hat, logL, exitflag, output, grad, hessian] = fminunc(log_likelihood, params_init, options);
se = sqrt(diag(inv(hessian))); % Standard errors
t_values = params_hat ./ se; % t-statistics
p_values = 2 * (1 - normcdf(abs(t_values))); % Two-tailed p-values

disp('Standard errors:');
disp(se);
disp('t-values:');
disp(t_values);
disp('p-values:');
disp(p_values);

