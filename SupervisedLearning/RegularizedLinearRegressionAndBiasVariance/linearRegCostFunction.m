function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute hypothesis (m x 1 vector)
% X = m x n matrix
% theta = n x 1 vector
hypothesis = X * theta;  % predictions of hypothesis

% Compute cost J (single value)
regularizationCost = lambda / (2 * m) * sum(theta(2:end).^2);
J = 1/(2*m) * (hypothesis - y)' * (hypothesis - y) + regularizationCost;


% Compute the gradient (n x 1 vector)
% X = m x n matrix
% y = m x 1 vector
regularizationGradient = (lambda / m) * [0; theta(2:end)];
grad = (1/m) * X' * (hypothesis - y) + regularizationGradient;


% =========================================================================

grad = grad(:);

end
