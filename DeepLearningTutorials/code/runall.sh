#!/usr/bin/env bash

EXCLUDE="DBN.py SdA.py cA.py dA.py rbm.py rnnrbm.py test.py utils.py"

FOLDER=result-$(date +"%y%m%d%H%M")

if [ -d $FOLDER ]; then
  NUM=0
  until [ ! -d $FOLDER-$NUM ]; do
    let "NUM = NUM + 1"
    echo $NUM
  done
  FOLDER=$FOLDER-$NUM
fi

mkdir $FOLDER

for ml in *.py; do
  skip=0
  for files in $EXCLUDE; do
    if [ "$ml" == "$files" ]; then
      skip=1
      continue
    fi
  done
  if [ $skip -ne 0 ]; then
    continue
  fi

  echo running $ml
  python ./$ml &> $FOLDER/${ml%.py}.out
done
