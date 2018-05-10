function y = normalize(x)
y = (x - min(x(:))) / (quantile(x(:),0.999) - min(x(:)));
end