#!/bin/bash

#BSUB -W 120:00
#BSUB -n 1
#BSUB -o euleroutputs/outfile_%J.%I.txt
#BSUB -R "rusage[mem=120000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=5500]"
#BSUB -R "rusage[scratch=120000]"
#BSUB -J "rs[1-5]"

# job index (set this to your system job variable e.g. for parallel job arrays)
# used to set model_idx and test_fold_idx below.
#index=0   # index=0 --> model_idx=0, test_fold_idx=0
index=$((LSB_JOBINDEX))
rs=$(( ($index % 5) + 1610 ))

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

# more regularization
#python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,nga,moz -lr 0.000001 -optim adam -wr 0.03 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_1all3
#python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,nga,moz -lr 0.000001 -optim adam -wr 0.03 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_2all3
#python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,nga,moz -lr 0.000001 -optim adam -wr 0.03 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_3all3
#python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,nga,moz -lr 0.000001 -optim adam -wr 0.03 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_4all3
#python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,nga,moz -lr 0.000001 -optim adam -wr 0.03 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_5all3
#python superpixel_disagg_model.py -train uga,rwa -train_lvl f,f,f,f,f -test tza,uga,rwa,nga -lr 0.000001 -optim adam -wr 0.03 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm d,d,d,d,d,d --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_X3 --e5f_metric best_mape_avg -e5f transfer_LogL1_${rs}_1all3,transfer_LogL1_${rs}_2all3,transfer_LogL1_${rs}_3all3,transfer_LogL1_${rs}_4all3,transfer_LogL1_${rs}_5all3


# without NGA,MOZ
#python superpixel_disagg_model.py -train uga,rwa,cod -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_1nm
#python superpixel_disagg_model.py -train uga,rwa,cod -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_2nm
#python superpixel_disagg_model.py -train uga,rwa,cod -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_3nm
#python superpixel_disagg_model.py -train uga,rwa,cod -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_4nm
#python superpixel_disagg_model.py -train uga,rwa,cod -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_5nm
#python superpixel_disagg_model.py -train uga,rwa -train_lvl f,f,f,f,f -test tza,uga,rwa -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm d,d,d,d,d --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_Xnm --e5f_metric best_mape_avg -e5f transfer_LogL1_${rs}_1nm,transfer_LogL1_${rs}_2nm,transfer_LogL1_${rs}_3nm,transfer_LogL1_${rs}_4nm,transfer_LogL1_${rs}_5nm

# without NGA
#python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_1n
#python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_2n
#python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_3n
#python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_4n
#python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_5n
python superpixel_disagg_model.py -train uga,rwa,moz -train_lvl f,f,f,f,f -test tza,uga -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm d,d,d,d,d --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_Xn --e5f_metric best_mape_avg -e5f transfer_LogL1_${rs}_1n,transfer_LogL1_${rs}_2n,transfer_LogL1_${rs}_3n,transfer_LogL1_${rs}_4n,transfer_LogL1_${rs}_5n
