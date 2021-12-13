#!/bin/bash

#BSUB -W 4:00
#BSUB -n 8
#BSUB -R "rusage[mem=4000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=5500]"
##BSUB -R "rusage[scratch=12500]"
#BSUB -J "popest"

# job index (set this to your system job variable e.g. for parallel job arrays)
# used to set model_idx and test_fold_idx below.
#index=0   # index=0 --> model_idx=0, test_fold_idx=0
index=$((LSB_JOBINDEX - 1))
val_fold=$(( $index % 5 ))

leave=Clipart

# cp -r /scratch2/Code/stylebias/data/OfficeHome $TMPDIR/
# cp -r /cluster/work/igp_psr/nkalischek/stylebias/data/OfficeHome $TMPDIR/
# cp -r /cluster/work/igp_psr/metzgern/HAC/code/codeJohn main/population_estimation/datasets $TMPDIR/

echo job index: $index
echo leave: $leave
echo val_fold: $val_fold

source HACenv/bin/activate

# load modules
module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1


python superpixel_disagg_model.py   -train uga,rwa,tza,nga,moz,cod \
                                    -train_lvl f,f,f,f,f,c \
                                    -test uga,rwa,tza,nga,moz,cod \
                                    -lr 0.0001 \
                                    -optim adam \
                                    -wr 0.001 \
                                    -adamwr 0. \
                                    -lstep 8000 \
                                    --validation_fold ${val_fold} \
                                    -mm m,m,m,m,d,d \
                                    --loss laplaceNLL \
                                    --custom_sampler_weights 1,1,1 \
                                    --input_scaling True \
                                    --output_scaling True

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
