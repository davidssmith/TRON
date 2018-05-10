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

function data = raread(filename)
fd = fopen(filename,'r');
h = getheader(fd);
ra_type_names = {'user','int','uint','float','complex'};
if h.eltype == 4 % complex float, which Matlab can't handle
    wascomplex = 1;
    h.elbyte = h.elbyte / 2;
    h.dims = [2; h.dims];
    h.eltype = 3;
else
    wascomplex = 0;
end
ratype = sprintf('%s%d',ra_type_names{h.eltype+1}, h.elbyte*8);
data = fread(fd, h.elbyte*prod(h.dims), ratype);
if ratype == 'float32'
    data = single(data);
end
if wascomplex
    tmp = complex(data(1:2:end), data(2:2:end));
    data = reshape(tmp, h.dims(2:end).');
else
    data = reshape(data, h.dims.');
end

function st = getheader(fd)
h = fread(fd, 6, 'uint64');  
st = {};
st.flags = h(2);
st.eltype = h(3);
st.elbyte = h(4);
st.size = h(5);
st.ndims = h(6);
st.dims = fread(fd, st.ndims, 'uint64');
