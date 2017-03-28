#!/bin/sh
make && 
{ 
  echo ------- degridding
  ./tron -v ../data/shepplogan.ra sl_data_tron.ra 
} && { 
  echo ------- re-gridding 
  ./tron -a -v sl_data_tron.ra sl_tron_tron.ra
}
./tron -a -v sl_data_irt.ra sl_irt_tron.ra


echo ANSWER: 11182.9795 
