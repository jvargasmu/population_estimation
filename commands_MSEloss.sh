#!/bin/bash

superpixel_disagg_model.py -train tza --validation_fold 0 --name runame-0
superpixel_disagg_model.py -train tza --validation_fold 1 --name runame-1
superpixel_disagg_model.py -train tza --validation_fold 2 --name runame-2
superpixel_disagg_model.py -train tza --validation_fold 3 --name runame-3
superpixel_disagg_model.py -train tza --validation_fold 4 --name runame-4
superpixel_disagg_model.py -train tza -e5f runame-1,runame-2,runame-3,runame-4 -e5fmt best_r2
