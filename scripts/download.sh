#!/usr/bin/env bash

id=$1
interval=$2

formats=$(yt-dlp -F "${id}")

cd dataset

function download_hr {
  # 4K 60 vp9
  if echo "${formats}" | cut -d' ' -f1 | grep 315; then 
    yt-dlp -f 315 "${id}" -o hr.webm    
  # 4K 30  vp9
  elif echo "${formats}" | cut -d' ' -f1 | grep 313; then 
    yt-dlp -f 313 "${id}" -o hr.webm    
  # 1080p 30 h264
  elif echo "${formats}" | cut -d' ' -f1 | grep 137; then 
    yt-dlp -f 137 "${id}" -o hr.webm    
  fi

  ffmpeg -i hr.webm -r 1/"${interval}" -vf scale=1920:1080 %06d_hr_"${id}".png
}

function download_lr {
  # 360p webm
  if echo "${formats}" | cut -d' ' -f1 | grep 243; then
    yt-dlp -f 243 "${id}" -o lr.webm
  fi

  ffmpeg -i lr.webm -r 1/"${interval}" %06d_lr_"${id}".png    
}

mkdir -p "${id}"
cd "${id}"

download_hr
# download_lr
