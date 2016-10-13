#!/bin/sh
make && ./tron -d 512 -r Dgfcsa  ../data/ex_whole_body.ra && od -f -N 8 -j 80 img_tron.ra

echo ANSWER: 11182.9795
