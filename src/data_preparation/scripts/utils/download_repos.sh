#!/usr/bin/env bash

# Get the input's absolute path
input="$(cd "$(dirname "$1")"; pwd -P)/$(basename "$1")"
echo "Reading a project list from $input"

mkdir -p ./data_preparation/raw_repos
cd ./data_preparation/raw_repos

while IFS= read -r line
do
    echo "$line"
    dir=${line:19:$((${#line} - 23))}
    GIT_TERMINAL_PROMPT=0 git clone --depth=1 $line ${dir//"/"/"."}
done < "$input"

# Create dataset spec
for repo in ./*; do
    echo $repo
    cd ./$repo
    echo $(git remote get-url origin) $(git rev-parse HEAD) >> ../../dataset.spec
    cd ..
done

#####
# Optionally, we could delete all non-Python files, but it doesn't make a huge difference.
###
#find . -type f ! -name "*.py" ! -name "*.pyi" -exec rm -f {} \;

#Remove symbolic links
#find . -type l -delete
#find . -type d -empty -delete