#!/bin/bash

files=$(find dataset_360_v1/ -type f -name "*_hr_*.png")

for file in "${files}"
do
  echo $file
  echo "----------"
done