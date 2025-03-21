# INF2008_G22
Skin Lesion Classification using EfficientNet and SVM
## Introduction
This document outlines the process of classifying skin lesions using deep learning techniques. The project utilizes the HAM10000 dataset, employs EfficientNetB0 for feature extraction, and leverages Support Vector Machines (SVM) and XGBoost for classification.
## 1. Environment Setup
### 1.1. Google Colab
This project is designed to run on Google Colab, a cloud-based platform offering free GPU access, which accelerates the training process.
### 1.2. Libraries
If runned locally, the following libraries are crucial for the project's execution. To run natively on Windows, there is support for Tensorflow only up to version 2.10.0 and the user needs python 3.9-3.10. Do note that to utilise GPU, the user has to install NVIDIA CUDA Toolkit 11.2 and cuDNN 8.1.
https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
https://developer.nvidia.com/rdp/cudnn-archive


Ensure they are installed in your environment using pip:

!pip install -r requirements.txt


### 1.3. Dataset
Download the HAM10000 dataset.
https://www.kaggle.com/datasets/nightfury007/ham10000-isic2018-raw/data
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

### 2.2. Data Preprocessing
### Splitting The Dataset
    train, test = train_test_split(metadata, test_size=0.2, stratify=metadata['dx'], random_state=42)
    train, val = train_test_split(train, test_size=0.2, stratify=train['dx'], random_state=42)


### Handling Class Imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train['dx']), y=train['dx'])
    class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}
    

### Label mapping
    label_mapping = {label: idx for idx, label in enumerate(sorted(train['dx'].unique()))}
    train['dx'] = train['dx'].map(label_mapping)
    val['dx'] = val['dx'].map(label_mapping)
    test['dx'] = test['dx'].map(label_mapping)


### Image Preprocessing
    def preprocess_image_and_mask(image_path, mask_path, target_size=(224, 224)):
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    image = cv2.resize(image, target_size) / 255.0


    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to load mask: {mask_path}")
    mask = cv2.resize(mask, target_size) / 255.0


    masked_image = np.multiply(image, np.expand_dims(mask, axis=-1))
    return masked_image


### Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )


    val_datagen = ImageDataGenerator(rescale=1.0/255.0)


### Image Generator
    def masked_image_generator(data, batch_size=32, target_size=(224, 224), augment=False):
        while True:
            for start in range(0, len(data), batch_size):
                batch_data = data.iloc[start:start+batch_size]
                batch_images, batch_labels = [], []


            for _, row in batch_data.iterrows():
                try:
                    masked_image = preprocess_image_and_mask(row['image_path'], row['mask_path'], target_size)
                    if augment:
                        masked_image = train_datagen.random_transform(masked_image)
                    batch_images.append(masked_image)
                    batch_labels.append(row['dx'])
                except FileNotFoundError:
                    continue  # Skip missing images


            yield np.array(batch_images), tf.keras.utils.to_categorical(batch_labels, num_classes=len(label_mapping))


## 2.3. Feature Extraction
### Extracting Deep Features From Images
Load EfficientNetB0 for feature extraction (trained model should be used if available)
        
        efficientnet_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        feature_extractor = Model(inputs=efficientnet_base.input, outputs=efficientnet_base.layers[-5].output)


### Preprocessing images
    def preprocess_image(image_path, target_size=(224, 224)):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image = cv2.resize(image, target_size) / 255.0  # Resize and normalize
        return np.expand_dims(image, axis=0)  # Add batch dimension


### Extract Features From Images

    def extract_features(data, batch_size=32):
        features, labels = [], []

        for start in range(0, len(data), batch_size):
            batch_data = data.iloc[start:start+batch_size]
            batch_images, batch_labels = [], []
    
    
            for _, row in batch_data.iterrows():
                try:
                    image = preprocess_image(row['image_path'])
                    batch_images.append(image)
                    batch_labels.append(row['dx'])
                except FileNotFoundError:
                    continue  # Skip missing images
    
    
            if batch_images:
                batch_images = np.vstack(batch_images)
                batch_features = feature_extractor.predict(batch_images)
                features.append(batch_features)
                labels.extend(batch_labels)
    
    
        return np.vstack(features).reshape(len(labels), -1), np.array(labels)


### Extracting Features
    def extract_masked_features(data, batch_size=32):
        features, labels = [], []
    
    
        for start in range(0, len(data), batch_size):
            batch_data = data.iloc[start:start+batch_size]
            batch_images, batch_labels = [], []
    
    
            for _, row in batch_data.iterrows():
                try:
                    masked_image = preprocess_image_and_mask(row['image_path'], row['mask_path'])
                    batch_images.append(np.expand_dims(masked_image, axis=0))  # Add batch dimension
                    batch_labels.append(row['dx'])
                except FileNotFoundError:
                    continue  # Skip missing images
    
    
            if batch_images:
                batch_images = np.vstack(batch_images)  # Stack images
                batch_features = feature_extractor.predict(batch_images)  # Extract deep features
                features.append(batch_features)
                labels.extend(batch_labels)
    
    
        return np.vstack(features).reshape(len(labels), -1), np.array(labels)


### Scaled Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features)
    X_test_scaled = scaler.transform(X_test_features)

## 3. Data Processing
### 3.1. SVM Training
Train an SVM classifier on the extracted features. Hyperparameter tuning is performed using GridSearchCV with different cross-validation strategies. Refer to the code for details on parameter grids and cross-validation values.
### 3.2. XGBoost Training
Train an XGBoost classifier on the extracted features. Refer to the code for details on XGBoost parameters and training procedure.
### 3.3. Evaluation
Evaluate trained models on the test set using metrics like accuracy, confusion matrix, and classification report. The code provides functions for generating these evaluation results.
### 3.4. Prediction
For predicting skin conditions on new images, refer to the code snippet provided in the "Prediction" section of the original response.
## Conclusion
This project demonstrates the use of EfficientNet and SVM/XGBoost for skin lesion classification. Following these steps would allow for users to utilize the models and reproduce the results and gain insights into the process. Ensure to replace placeholders with actual values.

