#!/bin/bash

#BSUB -W 24:00
#BSUB -n 1
#BSUB -o euleroutputs/outfile_%J.%I.txt
#BSUB -R "rusage[mem=120000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=5500]"
#BSUB -R "rusage[scratch=120000]"
#BSUB -J "do0_6"

# job index (set this to your system job variable e.g. for parallel job arrays)
# used to set model_idx and test_fold_idx below.
#index=0   # index=0 --> model_idx=0, test_fold_idx=0
index=$((LSB_JOBINDEX))
rseed=$(( $index % 5 ))

leave=Clipart

# cp -r /scratch2/Code/stylebias/data/OfficeHome $TMPDIR/
# cp -r /cluster/work/igp_psr/nkalischek/stylebias/data/OfficeHome $TMPDIR/
cp -r -v /cluster/work/igp_psr/metzgern/HAC/code/repocode/population_estimation/datasets $TMPDIR/

echo job index: $index
echo leave: $leave
echo val_fold: $rseed
echo TEMPDIR: $TMPDIR

source HACenv/bin/activate

# load modules
module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1


#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 -adamwr 0. -lstep 80000 --validation_fold 0 -rs 2 -mm m,m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets
python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 -adamwr 0. -lstep 40 --validation_fold 0 -rs 2 -mm m,m,m,m,m,m --loss NormL1 --silent_mode True --dataset_dir datasets
#python superpixel_disagg_model.py -train uga,rwa,tza,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,tza,nga,moz,cod -lr 0.0001 -optim adam -wr 0.01 -adamwr 0. -lstep 25 --validation_fold 0 -rs 2 -mm m,m,m,m,m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets

#python superpixel_disagg_model.py   -train uga,rwa,tza,nga,moz \
#-train_lvl f,f,f,f,f \
# $TMPDIR/datasets                                    -test uga,rwa,tza,nga,moz,cod \
#                                    -lr 0.0001 \
#                                    -optim adam \
#                                    -wr 0.01 \
#				    --dropout 0.6 \
#                                    -adamwr 0. \
#                                    -lstep 8000 \
#                                    --validation_fold 0 \
#                                    -rs ${rseed} \
#                                    -mm m,m,m,m,m \
#                                    --loss laplaceNLL \
#                                    --input_scaling True \
#                                    --output_scaling True \
#				    --silent_mode True \
#                                    --dataset_dir $TMPDIR/datasets



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
