#!/bin/bash
backbone=$1


export HF_TOKEN=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

cd CLAM
python extract_features_fp.py --data_h5_dir test2_patches \
--data_slide_dir test2_slides --csv_path test2_patches/df_features.csv \
--feat_dir test2_features/${backbone} --batch_size 128 \
--slide_ext .tiff --target_patch_size 224 --model_name ${backbone}