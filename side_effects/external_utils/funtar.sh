#!/usr/bin/env bash

for f in $1/*;  do if [ -d $f ]; then echo $f; for z in $f/*tar.gz; do tar -zxvf $z -C $f; done; fi; done