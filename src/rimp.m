function x = rimp(image, gold)
if nargin < 2
    gold = zeros(size(image));
end
image = squeeze(image);
r = real(image);
i = imag(image);
m = abs(image);
p = angle(image)/pi;
x = abs([r i m p (image-gold)]);