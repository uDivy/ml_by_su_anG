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

part1 = (((X*theta)-y).^2);
cost = (1/(2*m)) * sum(part1, 'all');
total = 0;
for j = 2:size(theta)
    total = total + (theta(j,1)^2);
end
reg_cost = cost + ((lambda/(2*m))*total);

J = reg_cost;

% =========================================================================
[row, n] = size(X);
for l = 1
    val = 0;
    for k = 1:row
        val0 = (X(k,:)*theta)-y(k,1);
        val1 = (val0)*X(k,l);
        val = val + val1;
    end 
    grad(l) = (val/row);
end
for l = 2:n
    val = 0;
    for k = 1:row
        val0 = (X(k,:)*theta)-y(k,1);
        val1 = (val0)*X(k,l);
        val = val + val1;
    end 
    grad(l) = (val/row) + (lambda*theta(l)/row);
end
    
dbstop if error

grad = grad(:);
end
