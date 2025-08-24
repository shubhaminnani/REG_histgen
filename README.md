# "HistVLM: Pathology report generation using vision language model from gigapixel whole slide images"

### ✨ **Our method stood second in Test Phase 1!!**

This repo contains the dataset, model weight, and source code for REG challenge from MICCAI 2025.

<!-- **PathVLM: Pathology report generation using vision language model from gigapixel whole slide images**\
*Shubham Innani, Michael Feldman, S*\
Paper: <https://arxiv.org/abs/2403.05396> -->

<!-- Link to our paper: [[arxiv]](https://arxiv.org/abs/2403.05396) -->
<!-- 
### Highlight of our work
- We present **HistVLM**, a multiple instance learning-empowered framework for histopathology report generation together pathology specific foundation models. 
- Inspired by diagnostic and report-writing workflows, HistGen features two delicately designed modules, aiming to boost report generation by aligning whole slide images (WSIs) and diagnostic reports from local and global granularity. 
- To achieve this, a local-global hierarchical encoder is developed for efficient visual feature aggregation from a region-to-slide perspective. Meanwhile, a cross-modal context module is proposed to explicitly facilitate alignment and interaction between distinct modalities, effectively bridging the gap between the extensive visual sequences of WSIs and corresponding highly summarized reports.  -->
<!-- 
### Methodology
![](methodology.png)
Overview of the proposed HistGen framework: (a) local-global hierarchical encoder module, (b) cross-modal context module, (c) decoder module, (d) transfer learning strategy for cancer diagnosis and prognosis. -->
<!-- 
## Table of Contents
- [Dataset, model weight, source code for paper "HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction"](#dataset-model-weight-source-code-for-paper-histgen-histopathology-report-generation-via-local-global-feature-encoding-and-cross-modal-context-interaction)
    - [Highlight of our work](#highlight-of-our-work)
    - [Methodology](#methodology)
  - [Table of Contents](#table-of-contents)
  - [TO-DO](#to-do)
  - [Prerequisite](#prerequisite)
  - [HistGen WSI-report dataset](#histgen-wsi-report-dataset)
  - [Preprocessing and Feature Extraction with Pre-trained DINOv2 ViT-L](#preprocessing-and-feature-extraction-with-pre-trained-dinov2-vit-l)
  - [HistGen WSI Report Generation Model](#histgen-wsi-report-generation-model)
    - [Training](#training)
    - [Inference](#inference)
    - [Transfer to Downstream Tasks](#transfer-to-downstream-tasks)
  - [Issues](#issues)
  - [License and Usage](#license-and-usage) -->

<!-- ## News
- **2024-12-18**: Tokenizer for HistGen is uploaded, better decoding capability is unlocked. Check modules.tokenizers for details.
- **2024-12-18**: Ground Truth Reports are further cleaned and uploaded. Check the HuggingFace Datasets for more details.
- **2024-12-18**: Baselines models are uploaded.
- **2024-11-12**: HistGen WSI-report dataset is available on HuggingFace Datasets! (Also the annotation files!)
- **2024-08-10**: Codes for feature extraction (CLAM) is uploaded.
- **2024-06-17**: Our paper is accepted by MICCAI2024! 🎉 -->

<!-- ## TO-DO
- [x] Release the source code for training and testing HistGen
- [x] Release the diagnostic report data
- [x] Release the DINOv2 ViT-L features of WSIs
- [x] Release model weights of pre-trained DINOv2 ViT-L feature extractor
- [x] Release the source code for WSI patching and feature extraction
- [ ] Update checkpoints of HistGen and merge into EasyMIL for cancer diagnosis and survival analysis tasks -->

## Prerequisite
Follow this instruction to create conda environment and install necessary packages:
```
git clone https://github.com/shubhaminnani/REG_histgen
cd HistGen
conda env create -f enviroment.yml
```

## Preprocessing and Feature Extraction with Hoptimus1 and UNI2

We have added a sample slide at CLAM/test2_slides to create patches and features from all the slides.

To extract features for all the slides below is the directory structure expected from the script
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


### WSI Preprocessing
In this work, we adpoted [CLAM](https://github.com/mahmoodlab/CLAM) for preprocessing and feature extraction. We uploaded the minimal viable version of CLAM to this repo. For installation guide, we recommend to follow the original instructions [here](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md). To conduct preprocessing, please run the following commands:
```
cd HistGen
cd CLAM
conda activate clam_latest
sh CLAM/scripts/patches_20x.sh
```

### Request Access
Request access to [Hoptimus1](https://huggingface.co/bioptimus/H-optimus-1) and [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) to extract features from their respective repository

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
To test the model, simply run the following commands:
```
cd HistGen
conda activate histgen
./test_wsi_report.sh optimus1
./test_wsi_report.sh uni2
```

Before you run the script, please set the path and other hyperparameters in `test_wsi_report.sh`. 

