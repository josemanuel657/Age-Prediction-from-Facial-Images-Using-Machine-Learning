# Age Prediction from Facial Images Using Machine Learning

This Jupyter Notebook project focuses on predicting age from facial images using the [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new). With applications in social media, security, and healthcare, the project explores traditional machine learning models that balance accuracy and computational efficiency.


<div style="display: flex; flex-direction: row;">
    <img src="https://github.com/user-attachments/assets/840b6949-a406-4c12-a7e1-9634a5743a50" alt="Image 1" width="45%" style="margin-right: 10px;"/>
    <img src="https://github.com/user-attachments/assets/fc8ad0cd-5c6e-407b-9f30-dd4b1bfe8b3d" alt="Image 2" width="45%"/>
</div>

## Project Overview
Age prediction based on facial features is a challenging task with applications in various domains. This project uses the UTKFace dataset, which contains 20,000+ labeled images with metadata on age, gender, and ethnicity. We employ traditional machine learning methods alongside data preprocessing and interpretability techniques to predict age while maintaining efficiency on standard hardware.

## Dataset
The [UTKFace dataset](https://www.kaggle.com/jangedoo/utkface-new) includes:
- **Image Size**: 200x200 pixels
- **Labels**: Age, Gender, Ethnicity
- **Preprocessing**: Grayscale conversion, contrast adjustment, PCA for dimensionality reduction

![image](https://github.com/user-attachments/assets/f0979f99-fe24-4ed6-b533-54c817200186)

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
   - **Data Summary**: Examined age, gender, and ethnicity distribution.
   - **Visual Inspection**: Displayed random samples and visualized age, gender, and ethnicity distributions to understand dataset characteristics.
   - **Statistical Analysis**: Calculated age-related metrics (mean, median, mode) and visualized age distribution across demographic groups.

### 2. Data Preprocessing
   - **Balancing**: Ensured equal representation across gender and age groups by augmenting data.
   - **Grayscale Conversion and CLAHE**: Reduced image complexity while highlighting age-related features.
   - **Normalization & Zero-Centering**: Scaled images to [0,1] and zero-centered to improve model performance.
   - **Dimensionality Reduction**: Applied PCA to reduce dimensionality from 40,000 pixels to 500 components, maintaining essential age-related features.

### 3. Model Selection and Training
   - **Regression Trees**: Simple, interpretable models, including Random Forest and XGBoost with tuning.
   - **Support Vector Regression (SVR)**: Tried linear, polynomial, RBF, and sigmoid kernels to capture non-linear relationships.
   - **K-Nearest Neighbors (KNN)**: Evaluated Euclidean and Manhattan distances with tuned k values.
   - **Multi-Layer Perceptrons (MLP)**: Tested configurations with ReLU activation, early stopping, and L2 regularization for efficient age estimation.
   - **Regularized Linear Models**: Ridge, Lasso, Elastic Net, and Bayesian Ridge regressions were employed to manage high-dimensionality and prevent overfitting.

### 4. Model Evaluation
   - **Evaluation Metrics**: Assessed models using MSE, RMSE, MAE, and R².
   - **Results Summary**: The optimized MLP with ReLU activation and early stopping achieved the lowest RMSE, balancing accuracy with computational demands.
   - **Comparison**: XGBoost and RBF SVR also demonstrated good performance but required higher computation than the optimized MLP.

### 5. Model Interpretability
   - **SHAP Values**: Used SHAP to understand feature importance, linking PCA components back to facial features influencing age prediction.
   - **PCA Feature Mapping**: Visualized the influence of individual pixels on age predictions, adding interpretability to model decisions.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/josemanuel657/Facial-Age-Prediction-ML.git
   ```

2. Navigate into the project directory and install dependencies:
   ```bash
   cd Facial-Age-Prediction-ML
   pip install -r requirements.txt
   ```

3. Create a folder called `dataset`:
   ```bash
   mkdir dataset
   cd dataset
   ```

4. Download the UTKFace dataset:
   You can download the UTKFace dataset either by using the `kagglehub` library or manually from the [Kaggle link](https://www.kaggle.com/datasets/jangedoo/utkface-new).
   
   - **Using `kagglehub`**: Install `kagglehub` and download the dataset directly.
     ```bash
     pip install kagglehub
     ```

     Then, in your code:
     ```python
     import kagglehub

     # Download the latest version of the UTKFace dataset
     path = kagglehub.dataset_download("jangedoo/utkface-new")

     print("Path to dataset files:", path)
     ```

   - **Manually from Kaggle**: Go to the [UTKFace dataset page](https://www.kaggle.com/datasets/jangedoo/utkface-new), download the files, and move them into the `dataset` folder.

## Results
The optimized MLP model performed best, achieving the lowest RMSE. This model's interpretability was enhanced through SHAP, revealing which facial features contributed most to age prediction.

![image](https://github.com/user-attachments/assets/0db60c63-3024-4c4d-9c33-195e51d60a32)
