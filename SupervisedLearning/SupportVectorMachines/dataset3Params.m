function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% Possible values of C and val
C_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% Matrix with errors
error_matrix = zeros(size(C_val), size(sigma_val));

for i = 1:length(C_val)
  for j = 1:length(sigma_val)

    % Populate test values of C and sigma
    C_hyp = C_val(i);
    sigma_hyp = sigma_val(j);

    % Compute model
    model = svmTrain(X, y, C_hyp, @(x1, x2) gaussianKernel(x1, x2, sigma_hyp));

    % Compute predictions
    predictions = svmPredict(model, Xval);

    % Save values in matrix
    error_matrix(i, j) = mean(double(predictions ~= yval));
  end
end

% Compute the minimum error in the matrix
[row_min row_min_index] = min(error_matrix);
[col_min col_min_index] = min(row_min);

% Get the corresponding values of C and sigma
C = C_val(row_min_index(col_min_index));
sigma = sigma_val(col_min_index);


% =========================================================================

end
