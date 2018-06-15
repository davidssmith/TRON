#!/bin/sh
make &&
{
  echo ------- re-gridding Shepp-Logan phantom from TRON data
  #./tron -a sl_data_tron.ra sl_tron_tron.ra
  ./tron -a output/sl_data_irt.ra output/sl_irt_tron.ra
  #./tron -a sl_data_gn.ra sl_gn_tron.ra
  #./tron -a sl_data_bart.ra sl_bart_tron.ra
  echo ------- re-gridding whole-body volume using TRON
  ./tron -v -u 0.4 -d 21 -a -G ../data/ex_whole_body.ra output/img_cmt_tron.ra
  echo ------- re-gridding linear-angle phantom volume using TRON
  ./tron -v -u 0.4 -d 21 -a ../data/cmt_phantom_la.ra output/img_laph_tron.ra
  echo ------- re-gridding optic nerve volume using TRON
  ./tron -v -u 0.4 -d 21 -a -G ../data/optic_nerve_002.ra output/img_on_tron.ra
  #./tron -v -u 0.4 -d 21 -a -G ../data/optic_nerve_002.ra output/img_on_tron.ra
  #./tron -v -u 0.4 -d 21 -a -G ../data/optic_nerve_002.ra output/img_on_tron.ra
  #echo ------- re-gridding swallowing volume using TRON
  ./tron -v -u 0.4 -d 21 -a -G ../data/swallowing/raw_012.ra output/img_sw_tron.ra
  #./tron -v -u 0.4 -d 21 -a -G ../data/swallowing/raw_012.ra output/img_sw_tron.ra
  #./tron -v -u 0.4 -d 21 -a -G ../data/swallowing/raw_012.ra output/img_sw_tron.ra
}
