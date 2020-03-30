#!/usr/bin/env bash

if [ -z "$1" ]
then
     echo "No argument supplied"
     exit 1
fi

if ! [ -f "$1" ]
then
     echo "File doesn't exist"
     exit 1
fi

# Get the input's absolute path
input="$(cd "$(dirname "$1")"; pwd -P)/$(basename "$1")"
echo "Reading a project list from $input"

mkdir -p /usr/data/raw_repos
cd /usr/data/raw_repos

# To clone a dataset based on a .spec file:
#   * Comment the lines below that clone the dataset and create the .spec
#   * Replace with:
#           bash /usr/src/datasetbuilder/scripts/clone_from_spec.sh path-to-spec-file.spec

while IFS= read -r line
do
    echo "$line"
    dir=${line:19:-4}
    GIT_TERMINAL_PROMPT=0 git clone --depth=1 $line ${dir//"/"/"."}
done < "$input"

# Create dataset spec
for repo in ./*; do
    cd ./$repo
    echo $(git remote get-url origin) $(git rev-parse HEAD) >> ../../dataset.spec
    cd ..
done

cd ..

###
# Run deduplication. This assumes that .NET Core is installed.
###

git clone --depth=1 https://github.com/microsoft/near-duplicate-code-detector.git
mkdir -p ./repo_tokens
python3 ./near-duplicate-code-detector/tokenizers/python/tokenizepythoncorpus.py ./raw_repos/ ./repo_tokens/
echo "In " $PWD
dotnet run --project ./near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj -- --dir="./repo_tokens/" "./corpus_duplicates"


###
# We are now ready to run pytype on our full corpus
##
export SITE_PACKAGES=/usr/local/lib/python3.6/dist-packages
for repo in ./raw_repos/*; do
     echo Running: pytype -V3.6 --keep-going -o ./pytype -P $SITE_PACKAGES:$repo infer $repo
     pytype -V3.6 --keep-going -o ./pytype -P $SITE_PACKAGES:$repo infer $repo

     files=$(find $repo -name "*.py")
     for f in $files
     do
         f_stub=$f"i"
         f_stub="./pytype/pyi"${f_stub#"$repo"}
         if [ -f $f_stub ]; then
             echo Running: merge-pyi -i $f $f_stub
             merge-pyi -i $f $f_stub
         fi
     done
 done

readonly SRC_BASE="/usr/src/datasetbuilder/scripts/"
export PYTHONPATH="$SRC_BASE"
mkdir -p graph-dataset
python3 "$SRC_BASE"graph_generator/extract_graphs.py ./raw_repos/ ./corpus_duplicates.json ./graph-dataset $SRC_BASE/../metadata/typingRules.json --debug
mkdir -p graph-dataset-split
python3 "$SRC_BASE"utils/split.py -data-dir ./graph-dataset -out-dir ./graph-dataset-split
