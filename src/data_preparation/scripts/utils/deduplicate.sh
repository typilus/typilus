#!/usr/bin/env bash

cd ./data_preparation
git clone --depth=1 https://github.com/microsoft/near-duplicate-code-detector.git
mkdir -p ./repo_tokens
python3 ./near-duplicate-code-detector/tokenizers/python/tokenizepythoncorpus.py ./raw_repos/ ./repo_tokens/
echo "In " $PWD
dotnet run --project ./near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj -- --dir="./repo_tokens/" "./corpus_duplicates"