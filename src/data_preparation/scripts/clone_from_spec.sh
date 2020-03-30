#!/bin/bash

export GIT_TERMINAL_PROMPT=0

while IFS=" " read -r URL TARGET_SHA
do
  echo "url:$URL  at $TARGET_SHA"
  project=${URL:19:-4}
  TARGET_DIR_NAME=${project//"/"/"."}
  mkdir $TARGET_DIR_NAME
  git clone $URL $TARGET_DIR_NAME
  cd $TARGET_DIR_NAME
    git checkout $TARGET_SHA
    git checkout -b typilus/pldi2020-state
  cd ..
done < $1
