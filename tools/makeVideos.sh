#!/bin/bash
if [ "$1" == '' ] || [ "$2" == '' ]; then
    echo "Usage: $0 <folder> <frame rate>";
    exit;
fi
for folder in "$1"/**; do
    filename="$(cut -d'/' -f3 <<<"$folder")".mp4
    ffmpeg -r "$2" -f image2 -i "$folder"/frame-%d.png -vcodec libx264 -crf 18 "$folder"/"$filename"
done