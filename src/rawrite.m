% This file is part of the RA package (http://github.com/davidssmith/ra).
%
% The MIT License (MIT)
%
% Copyright (c) 2015 David Smith
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function rawrite(data, filename, ntrailing)
if nargin < 3
    ntrailing = 0;
end
w = whos('data');
elbytes = w.bytes / numel(data);
if isinteger(data)
    if strfind(w.class, 'uint')
        eltype = uint64(2);
    else
        eltype = uint64(1);
    end
elseif isfloat(data)
    if w.complex
        eltype = 4;
        %elbytes = elbytes / 2;
        re = real(data);
        im = imag(data);
        data = zeros([2,size(data)]);
        data(1:2:end) = re;
        data(2:2:end) = im;
        clear re im;
    else
        eltype = 3;
    end
else
    eltype = 0;
end
f = fopen(filename,'w');
flags = uint64(0);
filemagic = uint64(8746397786917265778);
nd = ndims(data) + ntrailing;
if w.complex, nd = nd -1; end
dims = [size(data) ones(1,ntrailing)];
if w.complex, dims = dims(2:end); end
fwrite(f, [filemagic, flags, eltype, elbytes, w.bytes, nd, dims], 'uint64');
fwrite(f, data(:), w.class);
fclose(f);
