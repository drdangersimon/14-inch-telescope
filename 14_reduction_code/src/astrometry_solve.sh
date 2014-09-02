#!/bin/bash

for i in $(find $1);
do
echo $i
solve-field -v --no-background-subtraction --scale-units arcsecperpix --scale-low .6 --scale-high 2.88 --overwrite --dir $2 --no-plots $i --use-sextractor;
done
