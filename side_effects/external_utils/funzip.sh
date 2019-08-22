#!/usr/bin/env bash

for f in $1/*
do
  filename=$(basename -- "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  if [ $extension == "zip" ]
  then
      echo $f
      mkdir $2/$filename
      unzip $f -d $2/$filename
  fi
done