#!/bin/bash

# List of Commands

### ScaleNet - one country - coarse (& adj.)
# laplaceNLL, fold 0
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000 --name LogL1_c_${rs}_1
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000 --name LogL1_c_${rs}_2
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000 --name LogL1_c_${rs}_3
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000 --name LogL1_c_${rs}_4
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000 --name LogL1_c_${rs}_5
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 50000 --name LogL1_c_${rs}_X --e5f_metric best_mape -e5f LogL1_c_${rs}_1,LogL1_c_${rs}_2,LogL1_c_${rs}_3,LogL1_c_${rs}_4,LogL1_c_${rs}_5


### ScaleNet - one country - fine (& adj.)
# laplaceNLL, fold 0
#a
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name laplace_f_${rs}_1a
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name laplace_f_${rs}_2a
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name laplace_f_${rs}_3a
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name laplace_f_${rs}_4a
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name laplace_f_${rs}_5a
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -mm d --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f laplace_f_${rs}_1a,laplace_f_${rs}_2a,laplace_f_${rs}_3a,laplace_f_${rs}_4a,laplace_f_${rs}_5a


### ScaleNet - multi country - fine (& adj.)
# laplaceNLL, fold 0
python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 210000

### ScaleNet - transfer country - fine (& adj.)

python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,nga,moz,cod -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 40 --validation_fold 0 -rs 4 -mm m,m,m,m,m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets
python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,nga,moz,cod -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 40 --validation_fold 0 -rs 4 -mm m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets
python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,nga,moz,cod -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 40 --validation_fold 0 -rs 4 -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets
python superpixel_disagg_model.py -train uga,rwa -train_lvl f,f,f,f,f -test uga,rwa -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 400 --validation_fold 0 -rs 1610 -rsf 1610 -mm m,m,d,d,m --loss LogL1 --input_scaling True --output_scaling True --dataset_dir datasets --name transfer_to_tza
python superpixel_disagg_model.py -train uga,rwa,tza -train_lvl f,f,f,f,f -test tza -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 400 --validation_fold 0 -rs 1610 -rsf 1610 -mm d,d,d --loss LogL1 --input_scaling True --output_scaling True --dataset_dir datasets --name transfer_to_tza_evaled --eval_model transfer_to_tza --e5f_metric best_mape_avg

### Building Disaggregation - transfer country - fine (& adj.)
python building_disagg_baseline.py --output_dir output_dir/ --train_dataset_name uga,moz,rwa --test_dataset_name tza --global_disag 
