#!/bin/sh
make &&
{
  echo ------- degridding Shepp-Logan phantom using TRON
  ./tron ../data/shepplogan.ra output/sl_data_tron.ra
} 
