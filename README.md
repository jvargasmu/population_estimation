# Fine-grained Population Mapping from Coarse Census Counts and Open Geodata

This is the code repository to the paper "Fine-grained Population Mapping from Coarse Census Counts
and Open Geodata"

<p align="center">
  <img src="imgs/HAC_ Pop-Est Viz_v11.png" />
</p>
## Requirements
 Install the following packages
```bash
virtualenv -p python3 HACenv
source HACenv/bin/activate
pip install numpy torch matplotlib h5py wandb tqdm torchvision fiona sklearn gdal==3.2.1
```

## Data
The raw data can be accessed through websites of [WorldPop](https://hub.worldpop.org/project/categories?id=14) and [Googles's Open Building](https://sites.research.google/open-buildings/).
For your convinience, we provide the precompiled datasets [here](https://drive.google.com/drive/folders/18IIPzj0pBOkq9T7SL-UYXz2AZMXhOuuZ?usp=sharing).

## Population Maps

Our Population Maps can be accessed [here](https://drive.google.com/drive/folders/1w_DGqBW4SkPIferoeKKVQNx1MrmgHws-?usp=sharing])


## Running the code for Tanzania
for random seeds ```${rs} = [1610,1611,1612,1612,1614]```

### Fine coarse training setting
```bash
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 0 -rs 1610 -rsf 1610 -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 50000 --name TZA_coarse_1610_1
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 0 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 50000 --name TZA_coarse_${rs}_1
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 1 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 50000 --name TZA_coarse_${rs}_2
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 2 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 50000 --name TZA_coarse_${rs}_3
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 3 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 50000 --name TZA_coarse_${rs}_4
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 50000 --name TZA_coarse_${rs}_5
python superpixel_disagg_model.py -train tza -train_lvl c -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 4 -rs ${rs} -rsf ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 50000 --name TZA_coarse_${rs}_X --e5f_metric best_mape -e5f TZA_coarse_${rs}_1,TZA_coarse_${rs}_2,TZA_coarse_${rs}_3,TZA_coarse_${rs}_4,TZA_coarse_${rs}_5
```
 
### Fine fine training setting
```bash
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 0 -rs ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 150000 --name TZA_fine_${rs}_1al
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 1 -rs ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 150000 --name TZA_fine_${rs}_2al
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 2 -rs ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 150000 --name TZA_fine_${rs}_3al
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 3 -rs ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 150000 --name TZA_fine_${rs}_4al
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 4 -rs ${rs} -mm m --loss LogL1 --dataset_dir datasets --sampler custom --max_step 150000 --name TZA_fine_${rs}_5al
python superpixel_disagg_model.py -train tza -train_lvl f -test tza -wr 0.01 --dropout 0.4 -lstep 800 --validation_fold 0 -rs ${rs} -mm d --loss LogL1 --dataset_dir datasets --sampler custom --max_step 100000 --name TZA_fine_${rs}_Xal --e5f_metric best_mape -e5f TZA_fine_${rs}_1al,TZA_fine_${rs}_2al,TZA_fine_${rs}_3al,TZA_fine_${rs}_4al,TZA_fine_${rs}_5al

