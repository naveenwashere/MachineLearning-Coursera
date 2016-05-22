function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(size(X, 1), 1) X];
% Convert y from (1-10) class into num_labels vector
yVec = eye(num_labels);
y = yVec(y,:);

z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(1, size(a2,2)); a2];
z3 = a2' * Theta2';
a3 = sigmoid(z3);

calc = (-y .* log(a3)) - ((1 - y) .* log(1 - a3));

% Unregularized Cost Function
J = ((1 / m) .* sum(sum(calc)));

tempTheta1 = Theta1;
tempTheta2 = Theta2;

tempTheta1(:, 1) = 0;
tempTheta2(:, 1) = 0;

sumTheta1 = sum(sum(power(tempTheta1, 2), 2));
sumTheta2 = sum(sum(power(tempTheta2, 2), 2));

J = J + (lambda / (2 * m)) * (sumTheta1 + sumTheta2);

% Refer to this: https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/threads/QFnrpQckEeWv5yIAC00Eog

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for i = 1 : m,
	a1 = X(1, :); 											% 1 x 401
	z2 = a1 * Theta1'; 										% 1 x 25
	a2 = sigmoid(z2);										% 1 x 25
	a2 = [1 a2]; 											% 1 x 26
	z3 = a2 * Theta2'; 										% 1 x 10
	a3 = sigmoid(z3); 										% 1 x 10

	d3 = a3 - y(i); 										% 1 x 10
	d2 = (d3 * Theta2)' .* sigmoidGradient([1 z2])';		% 1 x 25

	delta1 = delta1 + d2(2:end) * a1;						% 25 x 401
	delta2 = delta2 + d3' * a2;								% 10 x 26
end;

Theta1_grad = (1 / m) * delta1 + (lambda / m) * tempTheta1;
Theta2_grad = (1 / m) * delta2 + (lambda / m) * tempTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
