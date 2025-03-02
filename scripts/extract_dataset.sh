#!/usr/bin/env bash

for file in *.MOV
do
  id=${file%.*}
  mkdir -p "${id}"
  ffmpeg -i $file -r 1 ${id}/%06d_hr_"${id}".png
done



