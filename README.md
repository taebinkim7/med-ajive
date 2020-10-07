# CBCS Project

This GitHub is used to share code and project progress of CBCS project 

## Project Plan

### State 0: Reproduce the Results in JOINT AND INDIVIDUAL ANALYSIS OF BC

This stage is done. Some notes:
- Link to the paper: https://arxiv.org/abs/1912.00434
- Original Code of the paper can be found at: https://github.com/idc9/breast_cancer_image_analysis
- Code has been cleaned and updated. New code can be found at this github.

### Stage 1: Data Collecting and Cleaning
Collect the core image data and gene expression data. Previously only PAM50 gene expression data are used. Immune gene expression data will be collected and added to the gene expression data.

Notes:
- Number of subjects for image data: 1191
- Number of genes to use: 100 (PAM50+IMMUNE50)

### Stage 2: Do AJIVE Analysis with Image Data and New Gene Expression Data

### Stage 3: Develop DNN Framework to Predict Cancer Subtype with Image Data and Gene Expression Data
The DNN framework takes image data and gene expression data as input and predicts the subtype label of the data.

### Stage 4: Interpretation of DNN Output
Based on the supervised task in Stage 3, heatmap of input images can be generated to detect ROI in images.
