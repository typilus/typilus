# Typilus + Graph generation

On top of Typilus, this repository can be used to generate augmented ASTs that were used in Typilus.

## Graph generation

### Graph structure

The graphs contain the following edges:
* `CHILD` &ndash; AST edges
* `NEXT` &ndash; edges connecting subsequent tokens in code
* `NEXT_USE` &ndash; next usage of a variable
* `LAST_LEXICAL_USE` &ndash; previous usage of a variable
* `OCCURRENCES_OF` &ndash; edges between occurrences of the same variable 
* `SUBTOKEN_OF` &ndash; edges from subtokens to their origin
* `COMPUTED_FROM` &ndash; edges that point to the origins of a variable
* `RETURNS_TO` &ndash; edges from return/yield statements to the function definition

Currently, there are no CFG edges.

### Graph extraction
* Go to `src/data_preparation/scripts`
* Run `python -m graph_generator.run -i {input_dir} -o {output_dir}`
* You can select output format with `-f {format}`. Currently, `dot` and `jsonl_gz` are supported

## [Original Typilus](https://github.com/typilus/typilus)

A deep learning algorithm for predicting types in Python. Please find a preprint [here](https://arxiv.org/abs/2004.10657).

This repository contains its implementation (`src/`) and experiments (`exp/`).


Please cite as:
```
@inproceedings{allamanis2020typilus,
  title={Typilus: Neural Type Hints},
  author={Allamanis, Miltiadis and Barr, Earl T and Ducousso, Soline and Gao, Zheng},
  booktitle={PLDI},
  year={2020}
}
```
