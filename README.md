# Second-Hand Car Price Prediction using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/R²_Score-97.78%25-brightgreen.svg)](#results)

## Project Overview

This project implements a robust car price prediction model that achieves **97.78% accuracy** (R² score) on test data. The model uses advanced feature engineering techniques, including polynomial features and interaction terms, combined with a regularized deep neural network architecture.

## Technologies Used

- **Python 3.8+**
- **TensorFlow 2.0** - Deep learning framework
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Kaggle API** - Dataset acquisition

## Dataset

The project uses the **Second Hand Used Cars Dataset** from Kaggle, which contains comprehensive information about used cars including:

- Vehicle specifications (HP, Torque, Top Speed)
- Economic factors (Economy rating, On-road prices)
- Usage metrics (Years, Kilometers driven)
- Condition and rating information

**Dataset Source**: [Second Hand Used Cars Data Set - Linear Regression](https://www.kaggle.com/datasets/mayankpatel14/second-hand-used-cars-data-set-linear-regression)

## Model Architecture & Methodology

### Feature Engineering Pipeline

The project implements sophisticated feature engineering to capture non-linear relationships:

#### 1. **Polynomial Features**

- **Quadratic features**: `feature²` for all numerical columns
- **Cubic features**: `feature³` for all numerical columns
- Captures non-linear price relationships

#### 2. **Interaction Features**

Strategic feature combinations to model real-world relationships:

- `years × km` - Usage intensity
- `hp × torque` - Engine performance
- `economy × top_speed` - Efficiency vs performance trade-off
- `on_road_old × on_road_now` - Price depreciation pattern

#### 2. **Interaction Features**

Strategic feature combinations to model real-world relationships:

- `years × km` - Usage intensity
- `hp × torque` - Engine performance
- `economy × top_speed` - Efficiency vs performance trade-off
- `on_road_old × on_road_now` - Price depreciation pattern

#### 3. **Feature Scaling**

Applied **Normalization** using `tf.keras.layers.Normalization`:

- Scaled all features
- Standardized `y` (price) using z-score normalization:

  $$
  y_{\text{scaled}} = \frac{y - \text{mean}(y)}{\text{std}(y)}
  $$

---

#### The neural network architecture consists of:

An Input Layer that receives the normalized features.

Three Dense Layers with ReLU activation, containing 128, 64, and 32 neurons respectively.
Dropout Layers (0.3 rate) after the first two dense layers to prevent overfitting.
A final Dense Layer with a single neuron for price prediction.

### Neural Network Architecture

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

- **Optimizer**: Adam (learning rate: 0.0005)
- **Loss Function**: Mean Absolute Error (MAE)
- **Regularization**: L2 regularization (0.001) + Dropout (0.3)
- **Early Stopping**: Patience of 20 epochs
- **Data Split**: 80% train, 10% validation, 10% test

## Results

### Model Performance Metrics

| Metric       | Value          |
| ------------ | -------------- |
| **R² Score** | **97.78%**     |
| **MAE**      | **$15,111.97** |
| **RMSE**     | **$19,115.59** |

### Visualizations

_[Training Loss & RMSE Curves]_
![Training Curves](https://i.ibb.co/jkNYXdxd/Screenshot-2025-06-13-122720.png)

This plot illustrates the model's learning progress, showing how both training and validation loss/RMSE decrease over epochs, eventually stabilizing due to early stopping.

_[Feature Correlation Heatmap]_
![Correlation Heatmap](https://i.ibb.co/7xpdvmCJ/hitmap.png)

The heatmap provides a visual representation of the correlation between different features, helping to understand relationships within the dataset.

_[Actual vs Predicted Prices]_
![Prediction Scatter Plot](https://i.ibb.co/twNqyRfH/Screenshot-2025-06-13-122919.png)

This scatter plot visually compares the model's predictions against the actual car prices. A close alignment of points along the red dashed line indicates high accuracy.

### Model Training

- Early stopping to prevent overfitting
- Validation monitoring
- Best weights restoration

### Feature Importance

The model leverages both original features and engineered features:

- **Polynomial features** capture non-linear price depreciation
- **Interaction terms** model complex relationships between car attributes
- **Regularization** prevents overfitting despite high feature dimensionality

### Performance Analysis

- The high R² score (97.78%) indicates excellent model fit
- Low MAE ($15,112) shows good prediction accuracy
- RMSE slightly higher than MAE suggests some outliers, but overall robust performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- **Kaggle** for providing the dataset
- **TensorFlow** team for the excellent deep learning framework
- **Open source community** for the amazing tools and libraries

## Contact

Feel free to connect with me:

- GitHub: [@Aditto Mahmood](https://github.com/adittomahmood)
- LinkedIn: [Aditto Mahmood](https://linkedin.com/in/adittomahmood)

---

_Built with ❤️ using Python and TensorFlow_
