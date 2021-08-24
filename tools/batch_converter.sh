#! /bin/bash

#the arg variables
srcExt=$1
destExt=$2
srcDir=$3
destDir=$4
opts=$5

#create the dir
mkdir -p "$destDir"

#iterate over files with srcExt in srcDir 
for filename in "$srcDir"/*.$srcExt; do
    basePath=${filename%.*}
    baseName=${basePath##*/}

    ffmpeg -i "$filename" $opts "$destDir"/"$baseName"."$destExt"
done

echo "[._.] Conversion from ${srcExt} to ${destExt} complete."


# bash ffmpeg_batch.sh gif mp4 $d $d '-movflags faststart -pix_fmt yuv420p -vf scale=trunc(iw/2)*2:trunc(ih/2)*2'