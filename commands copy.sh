#!/bin/bash

### ScaleNet - one country - fine (& adj.)
# MSE, fold 0

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold ${rs} -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --sampler custom --max_step 100000
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs 0 -mm m --loss mse --input_scaling True --output_scaling True --sampler custom --max_step 100000
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss mse --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f ,,,,
