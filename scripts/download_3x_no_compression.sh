#!/usr/bin/env bash

id=$1
interval=1

formats=$(yt-dlp -F "${id}")
out_dir="dataset_3x_v2/${id}"

function download_video {
  # 4K 60 vp9
  if echo "${formats}" | cut -d' ' -f1 | grep 315; then 
    yt-dlp -f 315 "${id}" -o "${out_dir}/hr.webm"
  # 4K 30 vp9
  elif echo "${formats}" | cut -d' ' -f1 | grep 313; then 
    yt-dlp -f 313 "${id}" -o "${out_dir}/hr.webm" 
  # 1080p 30 h264
  elif echo "${formats}" | cut -d' ' -f1 | grep 137; then 
    yt-dlp -f 137 "${id}" -o "${out_dir}/hr.webm"  
  fi

  ffmpeg -i "${out_dir}/hr.webm" -r 1/"${interval}" -vf scale=1920:1080 "${out_dir}/%06d_hr_${id}.png"
  ffmpeg -i "${out_dir}/hr.webm" -r 1/"${interval}" -vf scale=640:360 "${out_dir}/%06d_lr_${id}.png"
}

mkdir -p "${out_dir}"

download_video
