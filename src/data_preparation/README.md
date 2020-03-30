# Data Preparation

We utilise [Docker](https://www.docker.com/) to improve the replicability of
data extraction. The Docker container will

1. download Python repositories;
2. use [pytype](https://github.com/google/pytype) to infer the types of some
   unannotated terms;
3. build graphs from the partially annotated Python code;
4. split the graphs into three sets: training, validation, and testing.

> We expect this process to take a few days, mainly because pytype
> needs a lot of time to infer types across a large Python corpus and fully resolve a
> large number of dependencies. To quickly test the pipeline, we recommend using
> the `typedRepos-small.txt` in the `metadata` folder instead of the full
> `typedRepos.txt` used below.
>
> In our experiments we employed an Azure [F32s v2](https://docs.microsoft.com/en-us/azure/virtual-machines/fsv2-series?toc=/azure/virtual-machines/linux/toc.json&bc=/azure/virtual-machines/linux/breadcrumb/toc.json) VM.

## Usage

First, build the Docker image from the provided `Dockerfile`. Assuming that you
are in the directory of this README file, run:

```bash
docker build . -t typilus-env
```

Then, create a container from the built image, mount `/local/path` to
`/container/path`, and run _bash_ in the container:

```bash
docker run --rm -it -v /local/path/to/store/data:/usr/data typilus-env:latest bash
```

Finally, to clone the repositories, infer types, build the graphs for the deep
learning model, and split the data into folds, run:

```bash
bash scripts/prepare_data.sh metadata/typedRepos.txt
```

Alternatively, run
```bash
bash scripts/prepare_data_small.sh
```
to create a small, toy corpus.

This script writes all related data to `/local/path/to/store/data`. Specifically,

* `dataset.spec` contains the git URLs and SHAs of the downloaded repositories.
* `graph-dataset-split` contains the extracted graphs in chunked `.jsonl.gz` format,
   split into train-validation-test in a replicable manner.
* `graph-dataset/_type_lattice.json.gz` contains the statically inferred
   type lattice for the given programs.

To extract the exact data used in the PLDI 2020 submission, you need to clone the
repositories and checkout the SHA shown in `pldi2020-dataset.spec` which can be
found in this folder. To achieve that modify `prepare_data.sh` as discussed in
that file. 

> Cloning multiple repos at a specific SHA takes some time. We recommend that you first test the
  script on `pldi2020-dataset-sample.spec` that contains only one repository.

To view the `.json.gz` files, consider using:

```bash
zcat file.json.gz | python -m json.tool | less
```

or for viewing (for example) the first line of a `.jsonl.gz` file:

```bash
zcat file.jsonl.gz | head -n 1 | python -m json.tool | less
```
