#!/bin/bash

#BSUB -W 4:00
#BSUB -n 1
#BSUB -o euleroutputs/outfile_%J.%I.txt
#BSUB -R "rusage[mem=120000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=5500]"
#BSUB -R "rusage[scratch=1200]"
#BSUB -J "eval[1-5]"

# job index (set this to your system job variable e.g. for parallel job arrays)
# used to set model_idx and test_fold_idx below.
#index=0   # index=0 --> model_idx=0, test_fold_idx=0
index=$((LSB_JOBINDEX))
rs=$(( ($index % 5) + 1610 ))

leave=Clipart

# cp -r /scratch2/Code/stylebias/data/OfficeHome $TMPDIR/
# cp -r /cluster/work/igp_psr/nkalischek/stylebias/data/OfficeHome $TMPDIR/
# cp -r -v /cluster/work/igp_psr/metzgern/HAC/code/repocode/population_estimation/datasets $TMPDIR/

echo job index: $index
echo leave: $leave
echo val_fold: $rs
echo TEMPDIR: $TMPDIR

source HACenv/bin/activate

# load modules
module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1

python superpixel_disagg_model.py -train uga,rwa,cod,nga,moz -train_lvl f,f,f,f,f -test tza,uga,rwa,nga,moz -lr 0.000001 -optim adam -wr 0.01 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm d,d,d,d,d,d --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --max_step 20000 --dropout 0.4 --name transfer_LogL1_${rs}_Xdo --e5f_metric best_mape_avg -e5f transfer_LogL1_${rs}_1do,transfer_LogL1_${rs}_2do,transfer_LogL1_${rs}_3do,transfer_LogL1_${rs}_4do,transfer_LogL1_${rs}_5do

#python superpixel_disagg_model.py -train uga,rwa -train_lvl f,f,f,f,f -test tza,uga,rwa -lr 0.0001 -optim adam -wr 0.01 -adamwr 0. -lstep 80 -rs ${rs} -rsf ${rs} -mm d,d,d,d,d,d --loss LogL1 --input_scaling True --output_scaling True --e5f_metric best_mae_avg --name transfer_LogL1_${rs}_Xallo -e5f transfer_LogL1_${rs}_1allo,transfer_LogL1_${rs}_2allo,transfer_LogL1_${rs}_3allo,transfer_LogL1_${rs}_4allo,transfer_LogL1_${rs}_5allo

#python superpixel_disagg_model.py -train tza -train_lvl c -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 50000 --name LogL1_c_${rs}_X --e5f_metric best_mape -e5f LogL1_c_${rs}_1,LogL1_c_${rs}_2,LogL1_c_${rs}_3,LogL1_c_${rs}_4,LogL1_c_${rs}_5

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm d --loss LogL1 --input_scaling True --output_scaling True --silent_mode True --dataset_dir datasets --sampler custom --max_step 100000 --name LogL1_f_${rs}_Xal --e5f_metric best_mape -e5f LogL1_f_${rs}_1al,LogL1_f_${rs}_2al,LogL1_f_${rs}_3al,LogL1_f_${rs}_4al,LogL1_f_${rs}_5al


#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f pretty-dragon-2288,electric-firebrand-2282,gentle-eon-2281,fresh-sound-2283,ethereal-disco-2284
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.00001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f feasible-plant-2279,glorious-cosmos-2286,glamorous-sun-2285,breezy-water-2285,dauntless-forest-2279
#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.000001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm m --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f eager-brook-2289,sandy-shape-2292,breezy-energy-2293,warm-feather-2291,fluent-dust-2290

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.000001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f dazzling-yogurt-2273,comfy-dragon-2272,generous-glitter-2276,good-snow-2275,deep-sound-2277

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.00001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f revived-paper-2271,noble-silence-2268,olive-surf-2267,fanciful-haze-2274,dandy-vortex-2269

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.000001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss LogL2 --input_scaling True --output_scaling True --dataset_dir datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f copper-pond-2269,zany-wind-2264,wobbly-disco-2264,eternal-dragon-2266,hearty-serenity-2263

#python superpixel_disagg_model.py -train tza -train_lvl f -test tza -lr 0.0001 -optim adam -wr 0.01 --dropout 0.4 -adamwr 0. -lstep 800 --validation_fold 0 -rs 0 -mm d --loss l1 --input_scaling True --output_scaling True --dataset_dir $TMPDIR/datasets --sampler custom --max_step 100000 --e5f_metric best_mape -e5f usual-silence-2195,fanciful-serenity-2193,youthful-brook-2191,frosty-sponge-2194,revived-thunder-2195

