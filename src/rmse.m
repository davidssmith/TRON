function y = rmse(x,x0)
y = sqrt(norm(x(:)-x0(:)).^2 / length(x(:)));