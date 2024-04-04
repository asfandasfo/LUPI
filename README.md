


# LUPI
**Paper: [Learning using Privileged Information for Chemosensitivity Prediction in Ovarian Cancer Patients](https://openaccess.thecvf.com/content_CVPRW_2020/html/w54/Yaar_Cross-Domain_Knowledge_Transfer_for_Prediction_of_Chemosensitivity_in_Ovarian_Cancer_CVPRW_2020_paper.html)**
![CNN Architecture](cnn_architecture.png)

## Introduction
This project utilizes Whole Slide Images (WSIs) and gene expression data to predict the chemosensitivity in ovarian cancer patients. WSIs can be downloaded from [GDC Data Portal]((https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-OV%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Tissue%20Slide%22%5D%7D%7D%5D%7D). ). 
The IDs of each WSI used in this project are available in the `data_files/imgs.csv` file.

To train and validate the models, one requires tiles extracted from WSIs, gene expression profiles, and their respective labels for each patient.

## 1. Pre-processing
WSIs are large in size, making it difficult to process the entire image. Therefore, we extracted the top 3 tiles from each slide. Preprocessing code can be found [here](https://github.com/deroneriksson/python-wsi-preprocessing). This tutorial contains three files: `slide.py`, `filter.py`, and `tiles.py`.

Before executing `slide.py`, add the path to the WSIs directory and the directory where you want to save the tiles on line 32, 143, and 751. This file will create low-resolution images of each whole slide image. No changes are required in `filter.py`. In the `tiles.py` file, change the size of the tile to 1536x2048 and top tiles to 3 on lines 38, 39, and 40, respectively. After these modifications, execute the code to generate the top 3 tiles from each WSI. The sequence is `slide.py` -> `filter.py` -> `tiles.py`.

## 2. Stain Normalization
Tiles extracted from WSIs need to be stain normalized to remove color variations. To do this, update the local paths in `normalization.py` and execute it.

After the above two steps, normalized top three tiles for each patient will be obtained. Add these tile paths to the `imgs.csv` file.

## 3. Data Files
This folder contains the following three files:

   I. `imgs.csv`
   
   II. `genes.csv`
   
   III. `labels.csv`

### imgs.csv:
This file contains the path to tiles for each patient used in this project and needs to be updated as per your local path.

### genes.csv:
In this file, each column represents a patient’s gene profile, which is already preprocessed as discussed in the paper.

### labels.csv:
This file contains labels (-1, 1) for each patient. -1 represents chemo-resistant and +1 represents chemo-sensitive patients.

**NOTE:** `genes.csv` and `labels.csv` files have data in the same sequence as `imgs.csv` file, which have patient IDs that will be used later for training. Also, change the local path to the tiles directory in `imgs.csv` file as needed.

## Training and Validation
By now, you will have all the data required to train and validate the models. Each model will use the `imgs.csv`, `genes.csv`, `labels.csv`, and `loader.py` files to load the data, and `patch_extractor.py` to extract patches from tiles. Make sure you update the local paths in the `imgs.csv` file.

To train the model, simply execute the model file, e.g., `privileged_model.py`, which will load the gene profiles of each patient from `genes.csv` and labels from `labels.csv` for training and validation of the privileged space model. Also, one can use the `testing.py` file to utilize the already trained model.




