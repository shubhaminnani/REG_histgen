#!/bin/bash
cd CLAM/
python create_patches_fp.py --source test2_slides \
--step_size 224 --patch_size 224 --patch --seg --save_dir test2_patches \
--preset bwh_biopsy.csv --patch_level 0