#!/bin/bash

#BSUB -W 24:00
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

### ScaleNet - one country - fine (& adj.)
# laplaceNLL, fold 0
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_1
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 1 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_2
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 2 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_3
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 3 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_4
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs 0 -mm m --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 150000 -name laplace_f_${rs}_5
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss laplaceNLL --input_scaling True --output_scaling True --silent_mode True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f laplace_f_${rs}_1,laplace_f_${rs}_2,laplace_f_${rs}_3,laplace_f_${rs}_4,laplace_f_${rs}_5

