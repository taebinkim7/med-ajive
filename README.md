# Medical Image Analysis with AJIVE

# Instructions to run the code

### 1. Setup data directories

cbcs_joint/Paths.py has instructions for setting up the data directory.

### 2. Install code

Download the github repository with
```
git clone https://github.com/taebinkim7/med-ajive.git
```
Change the folder path in cbcs_joint/Paths.py to match the data directories on your computer.

Using using python 3.7.2., (e.g. `conda create -n med-ajive python=3.7.2`, `conda activate med-ajive`) install the package `cbcs_joint` and the other required packages with
```
bash setup.sh
```

Note that it is important to install torch 1.0.0 with CUDA 10.0.


### 3. Image patch feature extraction

```
python scripts/patch_feat_extraction.py
```

This step extracts CNN features from each image patch and may take a few hours. If a GPU is available it will automatically be used.

### 4. AJIVE analysis

```
python scripts/ajive_analysis.py
```

The AJIVE analysis runs in about 30 seconds, but the whole script may take a while due to data loading and saving large figures.

### 5. Image visualizations

```
python scripts/image_visualizations.py
```

This may take a couple of hours and the resulting saved figures are a couple of gigabytes.
