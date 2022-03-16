#!/bin/bash

# List of Commands

#### Preprocessing commands ####



#### TZA ####
### Building disaggregation


### WorldPop - trained with coarse census data (adj.)


### MRF (adj.)


### ScaleNet - one country - coarse (& adj.)
# laplaceNLL, fold 0
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 50000
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -mm m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss l1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000

### ScaleNet - one country - fine (& adj.)
# laplaceNLL, fold 0
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f ,,,,

python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss NormL1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss NormL1 --input_scaling True --output_scaling True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f rose-vortex-2192,worthy-wood-2187,major-frog-2190,sweet-mountain-2189,charmed-sky-2188


python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss l1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss l1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss l1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f usual-silence-2195,fanciful-serenity-2193,youthful-brook-2191,frosty-sponge-2194,revived-thunder-2195

python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss mse --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f ,,,,

python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss LogL1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f ,,,,

python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000



python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f pretty-dragon-2288,electric-firebrand-2282,gentle-eon-2281,fresh-sound-2283,ethereal-disco-2284
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.00001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f feasible-plant-2279,glorious-cosmos-2286,glamorous-sun-2285,breezy-water-2285,dauntless-forest-2279
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.000001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f eager-brook-2289,sandy-shape-2292,breezy-energy-2293,warm-feather-2291,fluent-dust-2290

python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss LogL1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f copper-pond-2269,zany-wind-2264,wobbly-disco-2264,eternal-dragon-2266,hearty-serenity-2263
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.00001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss LogL1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f revived-paper-2271,noble-silence-2268,olive-surf-2267,fanciful-haze-2274,dandy-vortex-2269
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.000001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss LogL1 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f dazzling-yogurt-2273,comfy-dragon-2272,generous-glitter-2276,good-snow-2275,deep-sound-2277


### ScaleNet - multi country - fine (& adj.)
# laplaceNLL, fold 0
python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 210000

### ScaleNet - transfer country - fine (& adj.)

python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 40 --validation_fold 0 -rs 4 -mm m,m,m,m,m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets
python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 40 --validation_fold 0 -rs 4 -mm m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets

### Building Disaggregation - transfer country - fine (& adj.)
python building_disagg_baseline.py --output_dir output_dir/ --train_dataset_name uga,moz,rwa --test_dataset_name tza --global_disag 
