#!/bin/sh
make && 
{ 
  echo ------- degridding
  ./tron ../data/shepplogan.ra shepplogan_data.ra 
} && { 
  echo ------- re-gridding 
  ./tron -a shepplogan_data.ra shepplogan_tron.ra
}
echo ANSWER: 11182.9795 
