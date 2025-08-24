# "HistVLM: Pathology report generation using vision language model from gigapixel whole slide images"

<!-- ### ✨ **Our method stood second in Test Phase 1!!** -->

This repo contains the dataset, model weight, and source code for REG challenge from MICCAI 2025.

## Prerequisite
Follow this instruction to create conda environment and install necessary packages:
```
git clone https://github.com/shubhaminnani/REG_histgen
cd HistGen
conda env create -f enviroment.yml
```

## Preprocessing and Feature Extraction with Hoptimus1

We have added a sample slide at CLAM/test2_slides to create patches and features from all the slides.

To extract features for all the slides below is the directory structure expected from the script

```
CLAM/
|-- test2_slides
|    |-- slide_1.svs
|    |-- slide_2.svs
|    ╵-- ...
|-- test2_patches
|        |-- slide_1.h5
|        |-- slide_2.h5
|        ╵-- ...
╵-- test2_features/pt_files
|        |-- slide_1.h5
|        |-- slide_2.h5
|        ╵-- ...
```

### WSI Preprocessing
In this work, we adpoted [CLAM](https://github.com/mahmoodlab/CLAM) for preprocessing and feature extraction. We uploaded the minimal viable version of CLAM to this repo. For installation guide, we recommend to follow the original instructions [here](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md). To conduct preprocessing, please run the following commands:
```
cd HistGen
cd CLAM
conda activate clam_latest
sh CLAM/scripts/patches_20x.sh
```

### Request Access
Request access to [Hoptimus1](https://huggingface.co/bioptimus/H-optimus-1) to extract features from their respective repository

### Feature Extraction
To extract features of WSIs, please run the following commands:
```
cd HistGen
cd CLAM
conda activate clam_latest
sh CLAM/scripts/features_20x.sh
```

## HistGen WSI Report Generation Model

### Inference
Download the trained model [HistVLM](https://indiana-my.sharepoint.com/:u:/g/personal/sinnani_iu_edu/EYk3bxQvhd1HjR8_pmQpIDEBmxjCsDsDt8aWHerEahE1yQ?e=XhhPed) and place this .pt file inside as results/optimus1_all/model_best.pth

To test the model, simply run the following commands:
```
cd HistGen
conda activate histgen
./test_wsi_report.sh optimus1
```

Before you run the script, please set the path and other hyperparameters in `test_wsi_report.sh`. 