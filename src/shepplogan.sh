#!/bin/sh
make && ./tron ../data/shepplogan.ra shepplogan_data.ra && ./tron -a shepplogan_data.ra shepplogan_tron.ra

echo ANSWER: 11182.9795 
