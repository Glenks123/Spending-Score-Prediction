function [theta, J_History] = gradientDescent(X, y, theta, alpha, num_iters)
  
   m = length(y);
   J_History = zeros(num_iters, 1);
   
   for iter = 1:num_iters
     
     error = (X*theta)-y;
     theta = theta - (alpha * ((1/m) * (X' * error)));
     
     % Save the cost J for every iteration
     J_History(iter) = computeCost(X, y, theta);
     
   endfor
  
  end