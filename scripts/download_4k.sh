#!/usr/bin/env bash

id=$1
interval=1

formats=$(yt-dlp -F "${id}")

mkdir -p dataset_4k
cd dataset_4k

function download_hr {
  # 4K 60 vp9
  if echo "${formats}" | cut -d' ' -f1 | grep 315; then 
    yt-dlp -f 315 "${id}" -o hr.webm    
  # 4K 30  vp9
  elif echo "${formats}" | cut -d' ' -f1 | grep 313; then 
    yt-dlp -f 313 "${id}" -o hr.webm
  fi

  ffmpeg -i hr.webm -r 1/"${interval}" %06d_hr_"${id}".png
}

mkdir -p "${id}"
cd "${id}"

download_hr
