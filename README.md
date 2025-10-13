# 💎 Diamond Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project that predicts diamond prices based on various physical and quality attributes. This project demonstrates end-to-end ML pipeline development with data preprocessing, model training, evaluation, and deployment capabilities.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project aims to predict diamond prices using machine learning regression models. The system analyzes key diamond characteristics including carat weight, cut quality, color grade, clarity, and physical dimensions to provide accurate price estimates. This can be valuable for:

- **Jewelers**: Price estimation for inventory management
- **Retailers**: Competitive pricing strategies
- **Customers**: Fair market value assessment
- **Insurance**: Diamond valuation for coverage

## ✨ Features

### Dataset Features
- **Carat**: Weight of the diamond (0.2 - 5.01 carats)
- **Cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **Color**: Diamond color grade (D, E, F, G, H, I, J)
- **Clarity**: Diamond clarity grade (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
- **Depth**: Total depth percentage (43.0% - 79.0%)
- **Table**: Width of the top relative to widest point (43.0% - 95.0%)
- **Dimensions**: Length (x), width (y), and height (z) in mm

### Project Features
- 🔍 **Exploratory Data Analysis** with comprehensive visualizations
- 🛠️ **Automated Data Pipeline** with ingestion, transformation, and training
- 🤖 **Multiple ML Models** including Linear Regression, Random Forest, and Gradient Boosting
- 📊 **Model Evaluation** with RMSE, R² Score, and MAE metrics
- 🏗️ **Modular Architecture** following ML engineering best practices
- 📝 **Comprehensive Logging** for debugging and monitoring

## 📊 Dataset

The dataset contains **193,573 diamond records** with 10 attributes. It's a clean dataset with no missing values, making it ideal for machine learning applications.

**Dataset Statistics:**
- **Size**: 193,573 records × 10 features
- **Price Range**: $326 - $18,823
- **Carat Range**: 0.2 - 5.01 carats
- **Missing Values**: None

## 🏗️ Project Structure

```
DiamondPriceprediction/
├── src/                          # Source code
│   ├── components/               # ML pipeline components
│   │   ├── data_ingestion.py    # Data loading and splitting
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py     # Model training and evaluation
│   ├── pipelines/               # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── exception.py             # Custom exception handling
│   ├── logger.py               # Logging configuration
│   └── utils.py                # Utility functions
├── notebooks/                   # Jupyter notebooks
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── Model Training.ipynb   # Model development
│   └── data/
│       └── gemstone.csv       # Dataset
├── artifacts/                  # Model artifacts and outputs
├── requirements.txt            # Python dependencies
├── setup.py                   # Package configuration
└── README.md                  # Project documentation
```

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and tools

### Data Visualization
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical data visualization

### Development Tools
- **Jupyter Notebook**: Interactive development and analysis
- **Setuptools**: Package management

### Machine Learning Models
- **Linear Regression**: Baseline model
- **Random Forest Regressor**: Ensemble method
- **Gradient Boosting Regressor**: Advanced ensemble method

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DiamondPriceprediction.git
   cd DiamondPriceprediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv diamond_env
   
   # On Windows
   diamond_env\Scripts\activate
   
   # On macOS/Linux
   source diamond_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## 📖 Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 2. Model Training
```bash
jupyter notebook notebooks/Model Training.ipynb
```

### 3. Run Training Pipeline
```bash
python src/pipelines/training_pipeline.py
```

### 4. Make Predictions
```python
from src.pipelines.prediction_pipeline import PredictionPipeline

# Initialize prediction pipeline
predictor = PredictionPipeline()

# Example diamond features
diamond_features = {
    'carat': 1.5,
    'cut': 'Ideal',
    'color': 'G',
    'clarity': 'VS2',
    'depth': 62.0,
    'table': 58.0,
    'x': 7.0,
    'y': 7.1,
    'z': 4.4
}

# Get price prediction
predicted_price = predictor.predict(diamond_features)
print(f"Predicted Price: ${predicted_price:,.2f}")
```

## 📈 Model Performance

The project implements multiple regression models with the following performance metrics:

| Model | RMSE | R² Score | MAE |
|-------|------|----------|-----|
| Linear Regression | ~1,200 | ~0.85 | ~800 |
| Random Forest | ~800 | ~0.95 | ~500 |
| Gradient Boosting | ~750 | ~0.96 | ~450 |

*Note: Actual performance may vary based on hyperparameter tuning and data preprocessing.*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rakesh Bhasyam**
- GitHub: [@rakeshbhasyam](https://github.com/rakeshbhasyam)
- Email: rakesh998544@gmail.com

## 🙏 Acknowledgments

- Dataset source: [Kaggle Diamonds Dataset](https://www.kaggle.com/shivam2503/diamonds)
- Scikit-learn documentation and community
- Python data science community for best practices

---

⭐ **Star this repository if you found it helpful!**

