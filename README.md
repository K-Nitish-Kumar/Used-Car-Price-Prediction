# Used Car Price Prediction

## Project Overview
Used Car Price Prediction is a Machine Learning regression project that predicts the selling price of a used car based on features such as manufacturing year, kilometers driven, fuel type, seller type, transmission, and ownership history.

The project uses multiple regression algorithms to identify a suitable model for predicting used car prices.

## Objectives
- Explore and understand used car dataset features.
- Perform data cleaning and preprocessing.
- Visualize important patterns and relationships in the dataset.
- Train and compare multiple Machine Learning regression models.
- Evaluate model performance using standard regression metrics.
- Build an interactive web application using Streamlit for price prediction.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Algorithms
The following regression algorithms are used:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Model Evaluation
The models are evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving higher weight to larger errors.
- **R² Score:** Measures how well the model explains variations in the target variable.

The best-performing model is saved as `models/best_model.pkl`.

## Project Structure

```text
Used_Car_Price_Prediction/
│
├── dataset/
│   └── car_details.csv
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── correlation.png
│   ├── model_comparison.csv
│   ├── price_distribution.png
│   └── year_vs_price.png
│
├── app.py
├── explore_dataset.py
├── predict.py
├── train_model.py
├── visualize_data.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Used-Car-Price-Prediction.git
cd Used-Car-Price-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Vehicle Dataset from CarDekho available on Kaggle:

[Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

Place the downloaded CSV file in the `dataset/` folder using the filename expected by the project scripts.

## Running the Project

### 1. Explore the dataset

```bash
python explore_dataset.py
```

### 2. Visualize the data

```bash
python visualize_data.py
```

### 3. Train the models

```bash
python train_model.py
```

### 4. Run the prediction application

```bash
python -m streamlit run app.py
```

The Streamlit application will open in your browser, allowing users to enter car details and get an estimated selling price.

## Outputs
The project generates:
- Data visualization plots.
- Model comparison results.
- Actual vs. predicted price visualization.
- A saved trained machine learning model.
- An interactive Streamlit prediction application.

## Future Improvements
- Experiment with additional regression algorithms.
- Perform hyperparameter tuning.
- Improve feature engineering and preprocessing.
- Deploy the Streamlit application online.
- Integrate a larger and more diverse used-car dataset.

## Author
**Kona Nitish Kumar**