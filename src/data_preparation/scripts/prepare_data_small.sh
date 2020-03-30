#!/usr/bin/env bash

mkdir -p /usr/data/raw_repos
cd /usr/data/raw_repos

bash /usr/src/datasetbuilder/scripts/clone_from_spec.sh /usr/src/datasetbuilder/pldi2020-dataset-sample.spec

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
