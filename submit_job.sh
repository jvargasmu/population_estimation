#!/bin/bash

#BSUB -W 24:00
#BSUB -n 1
#BSUB -o euleroutputs/outfile_%J.%I.txt
#BSUB -R "rusage[mem=120000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=5500]"
#BSUB -R "rusage[scratch=120000]"
#BSUB -J "rs"

# job index (set this to your system job variable e.g. for parallel job arrays)
# used to set model_idx and test_fold_idx below.
#index=0   # index=0 --> model_idx=0, test_fold_idx=0
index=$((LSB_JOBINDEX))
rs=$(( $index % 5 ))

leave=Clipart

# cp -r /scratch2/Code/stylebias/data/OfficeHome $TMPDIR/
# cp -r /cluster/work/igp_psr/nkalischek/stylebias/data/OfficeHome $TMPDIR/
cp -r -v /cluster/work/igp_psr/metzgern/HAC/code/repocode/population_estimation/datasets $TMPDIR/

echo job index: $index
echo leave: $leave
echo val_fold: $rs
echo TEMPDIR: $TMPDIR

source HACenv/bin/activate

# load modules
module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1

python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.6 -adamwr 0. -lstep 8000 --validation_fold 0 -rs ${rs} -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000

#Laplace F multi
#python superpixel_disagg_model.py -train tza -train_lvl f,f,f,f,f,f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000 -e5f breezy-flower-2044,decent-darkness-2048,pious-breeze-2042,genial-sound-2045,clean-cosmos-2045 --e5f_metric best_mape

#Norm F multi
#python superpixel_disagg_model.py -train tza -train_lvl f,f,f,f,f,f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000 -e5f glorious-field-2047,fresh-sea-2049,silvery-gorge-2043,amber-microwave-2050,vivid-aardvark-2053 --e5f_metric best_mape

# Laplace F multi
#python superpixel_disagg_model.py -train rwa -train_lvl f,f,f,f,f,f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000 -e5f breezy-flower-2044,decent-darkness-2048,pious-breeze-2042,genial-sound-2045,clean-cosmos-2045 --e5f_metric best_mape

# Norm F multi
#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 1000 --validation_fold ${rs} -rs 0 -mm m,m,m,m,m,m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000

#python superpixel_disagg_model.py -train rwa -train_lvl f,f,f,f,f,f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000 -e5f glorious-field-2047,fresh-sea-2049,silvery-gorge-2043,amber-microwave-2050,vivid-aardvark-2053 --e5f_metric best_mape

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000 -e5f glorious-field-2047,fresh-sea-2049,silvery-gorge-2043,amber-microwave-2050,vivid-aardvark-2053 --e5f_metric best_mape

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 210000 -e5f breezy-flower-2044,decent-darkness-2048,pious-breeze-2042,genial-sound-2045,clean-cosmos-2045 --e5f_metric best_mape

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets--sampler custom --max_step 210000 -e5f breezy-flower-2044,decent-darkness-2048,pious-breeze-2042,genial-sound-2045,clean-cosmos-2045 --e5f_metric best_mape

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm m,m,m,m,m,m --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets--sampler custom --max_step 210000 -e5f glorious-field-2047,fresh-sea-2049,silvery-gorge-2043,amber-microwave-2050,vivid-aardvark-2053 --e5f_metric best_mape

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm d --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 -e5f twilight-bee-2031,different-valley-2027,true-feather-2028,fiery-star-2029,legendary-rain-2030 --e5f_metric best_mape

#python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm d --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 50000 -e5f iconic-universe-2012,glorious-monkey-2018,giddy-dew-2012,efficient-pond-2012,glorious-plasma-2012 --e5f_metric best_mape

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm d --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 -e5f stilted-dawn-2026,balmy-jazz-2022,eager-disco-2021,cool-thunder-2023,lemon-bee-2025 --e5f_metric best_mape


#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs 0 -mm d --loss NormL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets--sampler custom --max_step 100000 -e5f twilight-bee-2031,different-valley-2027,true-feather-2028,fiery-star-2029,legendary-rain-2030 --e5f_metric best_mape


#python superpixel_disagg_model.py

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold ${rs} -rs 1611 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets -sampler custom --custom_sampler_weights 1,1,1,1,1,1000

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs ${rs} -mm m,m,m,m,m,m --input_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets # --sampler custom  --custom_sampler_weights 1,1,1,1,1,50

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold 0 -rs ${rs} -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 200000

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold ${rs} -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 1400000

#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz,cod -train_lvl f,f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 8000 --validation_fold ${rs} -rs 0 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 200000

#python superpixel_disagg_model.py   -train uga,rwa,tza,nga,moz,cod \
#                                    -train_lvl f,f,f,f,f,f \
#                                    -test uga,rwa,tza,nga,moz,cod \
#                                    -lr 0.0001 \
#                                    -optim adam \
#                                    -wr 0.01 \
#				    --dropout 0.4 \
#                                    -adamwr 0. \
#					-lstep 8000 \
#                                    --validation_fold ${rs} \
#                                    -rs 1610 \
#                                    -mm m,m,m,m,m,m \
#                                    --loss laplaceNLL \
#				    --input_scaling True \
#				    --silent_mode True \
#                                    --dataset_dir $TMPDIR/datasets \
#                                    --sampler custom \
#                                    --custom_sampler_weights 1,1,1,1,1,1000

# python3 train.py --optimizer ADAM \
#                  --scheduler MultiStepLR \
#                  --base_learning_rate 0.00001 \
#                  --max_epochs 400 \
#                  --num_outputs 65 \
#                  --num_workers 8 \
#                  --l2_lambda 5e-5 \
#                  --otswap True \
#                  --swap_prob 0.6 \
#                  --cl_weight 2.5 \
#                  --contrastivelearning True \
#                  --lr_steps 100 200 \
#                  --gamma 0.2 \
#                  --leave_out ${leave} \
#                  --model_idx ${model_idx} \
#                  --experiment office \
#                  --dataset OfficeHome \
#                  --data_dir_root $TMPDIR/OfficeHome/ \
#                  --model resnet18-ma \
#                  --num_gpus 1 \
#                  --debug_mode False \
#                  --out_dir tmp/OfficeHome_v3/
