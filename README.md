# INF2008_G22
Skin Lesion Classification using EfficientNet and SVM
## Introduction
This document outlines the process of classifying skin lesions using deep learning techniques. The project utilizes the HAM10000 dataset, employs EfficientNetB0 for feature extraction, and leverages Support Vector Machines (SVM) and XGBoost for classification.
## 1. Environment Setup
### 1.1. Google Colab
This project is designed to run on Google Colab, a cloud-based platform offering free GPU access, which accelerates the training process.
### 1.2. Libraries
If runned locally, the following libraries are crucial for the project's execution. Ensure they are installed in your environment using pip:

!pip install opencv-python==4.7.0.72

!pip install tensorflow==2.12.0

!pip install scikit-learn==1.2.2

!pip install seaborn==0.12.2

!pip install pandas==1.5.3

!pip install xgboost==1.7.6


### 1.3. Dataset
Download the HAM10000 dataset.
Organize the dataset into three folders within a folder named dataverse_files in your Google Drive. These folders should contain:
HAM10000_images_combined_600x450: Lesion images.
HAM10000_segmentations_lesion: Lesion segmentation masks.
HAM10000_metadata: CSV file containing image metadata.
from google.colab import drive
drive.mount('/content/drive')







## 2. Data Processing
### 2.1. Loading Data
Load the metadata using pandas:
metadata = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dataverse_files/HAM10000_metadata')

Create paths for images and masks:
metadata['image_path'] = metadata['image_id'].apply(lambda x: os.path.join(images_path, f"{x}.jpg"))
metadata['mask_path'] = metadata['image_id'].apply(lambda x: os.path.join(mask_path, f"{x}_segmentation.png"))




### 3.2. SVM Training
Train an SVM classifier on the extracted features. Hyperparameter tuning is performed using GridSearchCV with different cross-validation strategies. Refer to the code for details on parameter grids and cross-validation values.
### 3.3. XGBoost Training
Train an XGBoost classifier on the extracted features. Refer to the code for details on XGBoost parameters and training procedure.
### 3.4. Evaluation
Evaluate trained models on the test set using metrics like accuracy, confusion matrix, and classification report. The code provides functions for generating these evaluation results.
### 3.5 Prediction
For predicting skin conditions on new images, refer to the code snippet provided in the "Prediction" section of the original response.
##Conclusion
This project demonstrates the use of EfficientNet and SVM/XGBoost for skin lesion classification. Following these steps would allow for users to utilize the models and reproduce the results and gain insights into the process. Ensure to replace placeholders with actual values.

