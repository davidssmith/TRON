function [err, coeffs] = lmse (target_image, reference_image)
%%LMSE  Least mean square error when image scaling allowed.
%
% [ERROR COEFFICIENTS] = LMSE(IMG1, IMG2) scales IMG1 until the minimum
% mean square ERROR of the difference found, then returns that error and
% the scaling COEFFICIENTS.
%
% Written by David S. Smith, 17 Nov 2010.

target_image = double(target_image(:));
reference_image = double(reference_image(:));
A = ones(numel(target_image),2);
A(:,1) = target_image;
coeffs = A \ reference_image;
err = norm(coeffs(1)*target_image + coeffs(2) - reference_image) / ...
	numel(target_image);