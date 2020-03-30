#!/bin/bash

mkdir -p ./k-ablation

for k in 4 5 7 9 11 13 16 19 25 ; do
   for p in 0.01 0.05 0.1 ; do
        echo "python3 typilus/utils/test.py  ../pydeeptype-runs/graph2hybridmetric-2_graph2hybridmetric-2_model_best.pkl.gz ~/dpufiles/data/pydeeptype/vOct12/split/test/ ~/dpufiles/data/pydeeptype/vOct12/_type_lattice.json.gz data_preparation/metadata/typingRules.json > ./k-ablation/ablation.k.$k.p.$p.eval.json"
        #NUM_K_NN=$k DIST_TEMPERATURE=$p python3 typilus/utils/test.py  ../pydeeptype-runs/graph2hybridmetric-2_graph2hybridmetric-2_model_best.pkl.gz ~/dpufiles/data/pydeeptype/vOct12/split/test/ ~/dpufiles/data/pydeeptype/vOct12/_type_lattice.json.gz data_preparation/metadata/typingRules.json > ./k-ablation/ablation.k.$k.p.$p.eval.json &
   done
   wait
   for p in 0.25 0.5 0.75 1 ; do
        echo "python3 typilus/utils/test.py  ../pydeeptype-runs/graph2hybridmetric-2_graph2hybridmetric-2_model_best.pkl.gz ~/dpufiles/data/pydeeptype/vOct12/split/test/ ~/dpufiles/data/pydeeptype/vOct12/_type_lattice.json.gz data_preparation/metadata/typingRules.json > ./k-ablation/ablation.k.$k.p.$p.eval.json"
        #NUM_K_NN=$k DIST_TEMPERATURE=$p python3 typilus/utils/test.py  ../pydeeptype-runs/graph2hybridmetric-2_graph2hybridmetric-2_model_best.pkl.gz ~/dpufiles/data/pydeeptype/vOct12/split/test/ ~/dpufiles/data/pydeeptype/vOct12/_type_lattice.json.gz data_preparation/metadata/typingRules.json > ./k-ablation/ablation.k.$k.p.$p.eval.json &
   done
   wait
   for p in 1.5 2 3 5 ; do
        echo "python3 typilus/utils/test.py  ../pydeeptype-runs/graph2hybridmetric-2_graph2hybridmetric-2_model_best.pkl.gz ~/dpufiles/data/pydeeptype/vOct12/split/test/ ~/dpufiles/data/pydeeptype/vOct12/_type_lattice.json.gz data_preparation/metadata/typingRules.json > ./k-ablation/ablation.k.$k.p.$p.eval.json"
        #NUM_K_NN=$k DIST_TEMPERATURE=$p python3 typilus/utils/test.py  ../pydeeptype-runs/graph2hybridmetric-2_graph2hybridmetric-2_model_best.pkl.gz ~/dpufiles/data/pydeeptype/vOct12/split/test/ ~/dpufiles/data/pydeeptype/vOct12/_type_lattice.json.gz data_preparation/metadata/typingRules.json > ./k-ablation/ablation.k.$k.p.$p.eval.json &
   done
   wait
done
