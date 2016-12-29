#!/bin/sh
make && 
{ 
  echo ------- degridding
  ./tron -r 512 -p 512 ../data/shepplogan.ra sl_data_tron.ra 
} && { 
  echo ------- re-gridding 
  ./tron -a -d 512 sl_data_tron.ra sl_tron_tron.ra
}
./tron -a -d 512 sl_data_irt.ra sl_irt_tron.ra


echo ANSWER: 11182.9795 
