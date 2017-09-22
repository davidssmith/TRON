function [diff_image, err, coeffs] = lmsediff (target_image, reference_image, mask)
%%LMSEDIFF Difference image subject to least mean square error scaling.
%
% [DIFF ERROR COEFFICIENTS] = LMSEDIFF(IMG1,IMG2) returns the DIFF image
% between IMG1 and IMG2 when IMG1 is subject to the linear scaling
% COEFFICIENTS that produce the smallest mean square ERROR.
%
% [DIFF ERROR COEFFICIENTS] = LMSEDIFF(IMG1,IMG2,MASK) returns the DIFF
% image between IMG1 and IMG2 when IMG1 is subject to the linear scaling
% COEFFICIENTS that produce the smallest mean square ERROR looking only
% inside the MASK.
%
% Written by David S. Smith, 17 Nov 2010.

diff_image = zeros(size(reference_image));
if nargin == 3
  target_image = target_image(mask);
  reference_image = reference_image(mask);
end

[err, coeffs] = lmse(target_image, reference_image);

if exist('mask','var')
  diff_image(mask) = coeffs(1)*target_image + coeffs(2) - reference_image;
else
  diff_image = coeffs(1)*target_image + coeffs(2) - reference_image;
end