#!/bin/sh
make &&
{
  echo ------- degridding
  ./tron ../data/shepplogan.ra sl_data_tron.ra
}  && {
  echo ------- re-gridding
  ./tron -a sl_data_irt.ra  sl_irt_tron.ra
  ./tron -a sl_data_gn.ra   sl_gn_tron.ra
  ./tron -a -v sl_data_bart.ra sl_bart_tron.ra
  ./tron -a -v sl_data_tron.ra sl_tron_tron.ra
}

VIEWER=$HOME/git/ra/python/raview
#echo TRON DATA
#$VIEWER sl_data_tron.ra -l
#echo IRT DATA
#$VIEWER sl_data_irt.ra -l
#echo TRON-TRON
#$VIEWER sl_tron_tron.ra
#echo TRON-TRON nz
#$VIEWER sl_tron_tron_1.ra
#echo IRT-TRON
#$VIEWER sl_irt_tron.ra
##echo IRT-TRON nz
#$VIEWER sl_irt_tron_1.ra

#echo ANSWER: 11182.9795
