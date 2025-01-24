# Diamond Price Prediction

This repository contains a Machine Learning project aimed at predicting the price of diamonds based on various features such as carat, cut, color, clarity, and more. By leveraging regression models, we can estimate the diamond price with high accuracy.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Usage](#usage)
- [License](#license)

## Overview
The goal of this project is to predict the price of diamonds based on specific attributes using machine learning models. Accurate price prediction can assist jewelers, retailers, and customers in making informed decisions.

## Features
Key features of the dataset used for prediction include:
- **Carat**: Weight of the diamond
- **Cut**: Quality of the cut (e.g., Fair, Good, Very Good, Premium, Ideal)
- **Color**: Diamond color, from D (best) to J (worst)
- **Clarity**: Diamond clarity, ranging from I1 (inclusions) to IF (flawless)
- **Depth**: Total depth percentage
- **Table**: Width of the top of the diamond relative to the widest point
- **Dimensions**: Length, width, and height measurements of the diamond

## Dataset
The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/shivam2503/diamonds). It contains 53,940 records with 10 attributes.

### Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Splitting data into training and testing sets

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
- **Frontend**:
  - HTML
  - CSS
- **Backend**:
  - Flask
- **Modeling Techniques**:
  - Linear Regression
  - Random Forest Regression
  - Gradient Boosting Regression

## Project Workflow
1. **Data Exploration**:
   - Analyze the dataset to understand distributions, outliers, and correlations.
2. **Data Preprocessing**:
   - Clean and transform the dataset.
3. **Feature Engineering**:
   - Create new features and optimize existing ones.
4. **Model Training**:
   - Train multiple regression models and evaluate performance.
5. **Model Evaluation**:
   - Compare models using metrics like RMSE, R2 Score, and MAE.
6. **Deployment**:
   - Deploy the best-performing model as a web application using Flask.


## Usage
1. Run the Jupyter Notebook to explore the data and train models:
   ```bash
   jupyter notebook
   ```
2. To run the web app:
   ```bash
   python app.py
   ```
3. Use the app to input diamond features and get a price prediction.

### Web Application
- The web app is built using Flask for the backend.
- HTML and CSS are used for the frontend to create a user-friendly interface.
- Users can input diamond attributes via the web interface and receive predicted prices instantly.



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author**: [Rakesh Bhasyam](https://github.com/rakeshbhasyam)

