#!/bin/sh
make && ./tron -a -d 512 ../data/ex_whole_body.ra && od -f -N 8 -j 80 img_tron.ra

echo ANSWER: 11182.9795