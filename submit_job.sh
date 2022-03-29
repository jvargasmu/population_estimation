#!/bin/bash

<<<<<<< HEAD
#BSUB -W 96:00
=======
#BSUB -W 120:00
>>>>>>> main
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
<<<<<<< HEAD
rs=$(( ($index % 5)+1610 ))
=======
rs=$(( ($index % 5) + 1610 ))
>>>>>>> main

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

<<<<<<< HEAD
# Kernel size kernel_size 3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_1_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_2_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_3_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_4_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_5_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_X_k3111 --e5f_metric best_mape -e5f LogL1_f_${rs}_1_k3111,LogL1_f_${rs}_2_k3111,LogL1_f_${rs}_3_k3111,LogL1_f_${rs}_4_k3111,LogL1_f_${rs}_5_k3111


# Kernel size kernel_size 5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_1_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_2_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_3_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_4_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_5_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_X_k5555 --e5f_metric best_mape -e5f LogL1_f_${rs}_1_k5555,LogL1_f_${rs}_2_k5555,LogL1_f_${rs}_3_k5555,LogL1_f_${rs}_4_k5555,LogL1_f_${rs}_5_k5555


# Kernel size kernel_size 3333
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_1_k3333
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_2_k3333
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_3_k3333
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_4_k3333
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_5_k3333
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_X_k3333 --e5f_metric best_mape -e5f LogL1_f_${rs}_1_k3333,LogL1_f_${rs}_2_k3333,LogL1_f_${rs}_3_k3333,LogL1_f_${rs}_4_k3333,LogL1_f_${rs}_5_k3333


# Kernel size kernel_size 3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_1_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_2_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_3_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_4_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_5_k3111
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,1,1,1 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_X_k3111 --e5f_metric best_mape -e5f LogL1_f_${rs}_1_k3111,LogL1_f_${rs}_2_k3111,LogL1_f_${rs}_3_k3111,LogL1_f_${rs}_4_k3111,LogL1_f_${rs}_5_k3111

# Kernel size kernel_size 5555
#3#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_1_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_2_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_3_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_4_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_5_k5555
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 5,5,5,5 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_X_k5555 --e5f_metric best_mape -e5f LogL1_f_${rs}_1_k5555,LogL1_f_${rs}_2_k5555,LogL1_f_${rs}_3_k5555,LogL1_f_${rs}_4_k5555,LogL1_f_${rs}_5_k5555


# Kernel size kernel_size 3333
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_1_k3333
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_2_k3333
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_3_k3333
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_4_k3333
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_5_k3333
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam --kernel_size 3,3,3,3 -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMDIR/datasets --sampler custom --max_step 150000 --name LogL1_f_${rs}_X_k3333 --e5f_metric best_mape -e5f LogL1_f_${rs}_1_k3333,LogL1_f_${rs}_2_k3333,LogL1_f_${rs}_3_k3333,LogL1_f_${rs}_4_k3333,LogL1_f_${rs}_5_k3333


### ScaleNet - one country - fine (& adj.)
# laplaceNLL, fold 0
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_1
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_2
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_3
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_4
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_5
#ipython superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f laplace_f_${rs}_1,laplace_f_${rs}_2,laplace_f_${rs}_3,laplace_f_${rs}_4,laplace_f_${rs}_5
=======
>>>>>>> main

# without NGA
python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_1n
python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_2n
python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_3n
python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_4n
python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test uga,rwa,moz,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m,m,m,m,m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_5n
python superpixel_disagg_model.py -train uga,rwa,cod,moz -train_lvl f,f,f,f,f -test tza,uga,rwa,cod,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm d,d,d,d,d --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --max_step 30000 --name transfer_LogL1_${rs}_Xn --e5f_metric best_mape_avg -e5f transfer_LogL1_${rs}_1n,transfer_LogL1_${rs}_2n,transfer_LogL1_${rs}_3n,transfer_LogL1_${rs}_4n,transfer_LogL1_${rs}_5n
