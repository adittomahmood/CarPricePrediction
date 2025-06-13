# Second-Hand Car Price Prediction using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-306998?style=for-the-badge&logo=python&logoColor=fff&labelColor=14213d&color=306998&logoWidth=20&shadow=1)](https://www.python.org/)
[![TensorFlow 2.0](https://img.shields.io/badge/TensorFlow-2.0-306998?style=for-the-badge&logo=tensorflow&logoColor=fff&labelColor=14213d&color=306998&logoWidth=20&shadow=1)](https://www.tensorflow.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-306998?style=for-the-badge&logo=open-source-initiative&logoColor=fff&labelColor=14213d&color=306998&shadow=1&logoWidth=20)](LICENSE)


## Project Overview

This project implements a robust car price prediction model that achieves an outstanding **97.78% accuracy** (R² score) on test data. The model utilizes advanced feature engineering techniques, including polynomial features and interaction terms, seamlessly integrated with a regularized deep neural network architecture. This approach ensures high prediction accuracy while effectively handling complex, non-linear relationships within the dataset.

## Technologies Used

The core of this project is built upon a powerful stack of modern data science and machine learning technologies:

* **Python 3.8+**: The primary programming language for data processing, model development, and analysis.
* **TensorFlow 2.0**: A leading open-source machine learning framework used for building and training the deep neural network.
* **Pandas**: Essential for efficient data manipulation and analysis.
* **Matplotlib**: Used for creating insightful visualizations of data distributions, model performance, and feature relationships.
* **Kaggle API**: Facilitates streamlined acquisition of the dataset.

## Dataset

The model is trained on the comprehensive **Second Hand Used Cars Dataset** sourced from Kaggle. This dataset provides a rich collection of attributes crucial for accurate car price prediction, including:

* **Vehicle specifications**: Horsepower (HP), Torque, Top Speed
* **Economic factors**: Economy rating, On-road prices (old and new)
* **Usage metrics**: Years since manufacture, Kilometers driven
* **Condition and rating information**: Various metrics describing the car's state and performance.

[![Dataset Source](https://img.shields.io/badge/Dataset_Source-blue?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mayankpatel14/second-hand-used-cars-data-set-linear-regression)

## Model Architecture & Methodology

The project employs a meticulously designed methodology, combining sophisticated feature engineering with a robust deep learning architecture to achieve high predictive performance.

### Feature Engineering Pipeline

To capture intricate non-linear relationships and improve model generalization, a comprehensive feature engineering pipeline was developed:

#### 1. Polynomial Features

* **Quadratic features ($feature^2$)**: Generated for all numerical columns to account for non-linear trends.
* **Cubic features ($feature^3$)**: Further enhance the model's ability to capture complex non-linear price relationships.

#### 2. Interaction Features

Strategic combinations of existing features were created to model real-world dependencies and amplify predictive power:

- `years × km` - Usage intensity
- `hp × torque` - Engine performance
- `economy × top_speed` - Efficiency vs performance trade-off
- `on_road_old × on_road_now` - Price depreciation pattern

#### 3. Feature Scaling

All features underwent **Normalization** using `tf.keras.layers.Normalization` to bring them to a similar scale, which is crucial for the stability and performance of neural networks. The target variable ($y$, price) was also standardized using z-score normalization:

$$y_{\text{scaled}} = \frac{y - \text{mean}(y)}{\text{std}(y)}$$

### Neural Network Architecture

The deep neural network is carefully structured to process the engineered features and provide accurate price predictions:

```
Input Layer (40+ features)
    ↓
Normalization Layer
    ↓
Dense Layer (128 units, ReLU) + L2 Regularization
    ↓
Dropout (30%)
    ↓
Dense Layer (64 units, ReLU) + L2 Regularization
    ↓
Dropout (30%)
    ↓
Dense Layer (32 units, ReLU) + L2 Regularization
    ↓
Output Layer (1 unit)
```

### Training Configuration

The model was trained with the following configurations to ensure optimal performance and prevent overfitting:

* **Optimizer**: Adam (with a learning rate of 0.0005)
* **Loss Function**: Mean Absolute Error (MAE), chosen for its robustness to outliers.
* **Regularization**: L2 regularization ($\lambda=0.001$) applied to dense layers, coupled with Dropout (0.3 rate) after the first two dense layers, to effectively combat overfitting, especially given the increased dimensionality from feature engineering.
* **Early Stopping**: Implemented with a patience of 20 epochs, monitoring validation loss to stop training when performance on unseen data no longer improves, and restoring the best weights.
* **Data Split**: The dataset was divided into 80% for training, 10% for validation, and 10% for testing, ensuring a robust evaluation of the model's generalization capabilities.

## Results

The model's performance was rigorously evaluated, demonstrating high accuracy and reliability in predicting second-hand car prices.

### Model Performance Metrics

The following metrics summarize the model's exceptional performance on the test set:

| Metric       | Value          |
| :----------- | :------------- |
| **R² Score** | **97.78%** |
| **MAE** | **$15,111.97** |
| **RMSE** | **$19,115.59** |

* The **R² score of 97.78%** indicates that the model explains nearly all the variance in car prices, signifying an excellent fit.
* A **Mean Absolute Error (MAE) of $15,111.97** suggests that, on average, the model's predictions are very close to the actual prices.
* The **Root Mean Squared Error (RMSE) of $19,115.59** is slightly higher than MAE, which is expected and suggests that while the model is generally accurate, there might be a few larger prediction errors for some outliers. Overall, it indicates robust performance.

### Visualizations

Visualizations provide deeper insights into the model's training process and predictive accuracy:

#### Training Curves

![Training Curves](https://i.ibb.co/jkNYXdxd/Screenshot-2025-06-13-122720.png)
This plot dynamically illustrates the model's learning progress. It shows the steady decrease in both training and validation loss/RMSE over epochs, eventually stabilizing due to the implemented early stopping mechanism, which prevents overfitting.

#### Correlation Heatmap

![Correlation Heatmap](https://i.ibb.co/Y4FT4sp6/Screenshot-2025-06-13-130438.png)
The heatmap provides a comprehensive visual representation of the correlation coefficients between different features in the dataset. This helps in understanding the linear relationships and dependencies, informing future feature engineering decisions.

#### Prediction Scatter Plot

![Prediction Scatter Plot](https://i.ibb.co/twNqyRfH/Screenshot-2025-06-13-122919.png)

This scatter plot offers a crucial visual comparison between the model's predicted car prices and the actual prices. A tight alignment of points along the red dashed line (representing perfect prediction) unequivocally indicates the model's high accuracy and strong predictive power.


## Contributing

Contributions are always welcome! If you have ideas for improvements, new features, or bug fixes, please feel free to submit a Pull Request. For significant changes or new functionalities, it is recommended to open an issue first to discuss the proposed changes.


## Acknowledgments

This project benefited immensely from the contributions and resources provided by:

* **Kaggle**: For hosting and providing the valuable dataset used in this project.
* **TensorFlow Team**: For developing an outstanding and versatile deep learning framework.
* **Open Source Community**: For the continuous development of incredible tools and libraries that make projects like this possible.

<div align="center">
  <h2>🌐 LET'S CONNECT & COLLABORATE</h2>
  <a href="https://linkedin.com/in/adittomahmood" target="_blank">
    <img src="https://custom-icon-badges.demolab.com/badge/_LinkedIn-0077B5?style=for-the-badge&logoColor=white&logo=linkedin" alt="LinkedIn Profile" />
  </a>
  <a href="https://github.com/adittomahmood" target="_blank">
    <img src="https://custom-icon-badges.demolab.com/badge/_Follow_Me-181717?style=for-the-badge&logoColor=white&logo=github" alt="GitHub Profile" />
  </a>
  <br/>
  <br/>
  <p style="font-size: 12px; color: #8B949E;">
    © 2025 Tasneem Bin Mahmood • Machine Learning & Computer Vision Engineer
  </p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/Trilokia/Trilokia/379277808c61ef204768a61bbc5d25bc7798ccf1/bottom_header.svg" width="100%" alt="Footer Image" />
</div>
