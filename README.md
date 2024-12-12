# Where's Dog, Where's Cat  
**Image Classification with Convolutional Neural Network (CNN)**  

## Key Features  
- **Image Preprocessing and Augmentation**  
  - Utilized `ImageDataGenerator` for preprocessing, including:  
    - Rescaling  
    - Shear transformations  
    - Zoom  
    - Rotation  
    - Horizontal/vertical shifts  
    - Brightness adjustments  
  - These augmentations enhance the model's generalization ability.

- **CNN Architecture**  
  - Three convolutional layers with increasing filter sizes.  
  - Max-pooling layers after each convolutional layer.  
  - Fully connected dense layer for binary classification.  

- **Regularization**  
  - Dropout layers to reduce overfitting.  
  - EarlyStopping monitors validation loss to halt training if overfitting occurs.  
  - `ReduceLROnPlateau` dynamically reduces the learning rate if validation loss plateaus.  

- **Training & Evaluation**  
  - Optimized using **Adam optimizer** with **binary cross-entropy loss**.  
  - Evaluated the model's performance on a separate test set.  

---

## Model Details  
- **Input Shape**: `(150, 150, 3)` (RGB images of size 150x150)  
- **Architecture**:  
  - 3 convolutional layers with filter sizes: `32`, `64`, `128`  
  - Max-pooling layers after each convolutional layer  
  - 1 dense layer with **64 units** and a **Dropout rate** of `0.3`  
  - Final output layer with **sigmoid activation** for binary classification  

- **Callbacks**:  
  - **EarlyStopping**: Stops training if validation loss doesn't improve for `5` consecutive epochs.  
  - **ReduceLROnPlateau**: Reduces learning rate by `0.5` if validation loss stagnates.  

---

## Requirements  
- TensorFlow (2.x)  
- Keras  
- NumPy  
- Matplotlib (for visualization)  

## Dataset  
The dataset for this project was sourced from [Kaggle](https://www.kaggle.com).  



