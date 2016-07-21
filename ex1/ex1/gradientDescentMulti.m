function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    % 

    tmp1 = 0;
    tmp2 = 0;
    tmp3 = 0;
    
    for i = 1:m
        tmp1 = tmp1 + (theta' * X(i,:)' - y(i));
        tmp2 = tmp2 + (theta' * X(i,:)' - y(i))*X(i,2);
        tmp3 = tmp3 + (theta' * X(i,:)' - y(i))*X(i,3);

    end
    theta(1) = theta(1) - tmp1*(alpha/m);
    theta(2) = theta(2) - tmp2*(alpha/m);
    theta(3) = theta(3) - tmp3*(alpha/m);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
