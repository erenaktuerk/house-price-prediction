House Price Prediction

This project is a professional and practice-oriented approach to predict house prices based on multiple features such as square footage, location, and condition. By using advanced machine learning techniques, the goal is to build a highly optimized and interpretable regression model that delivers accurate and reliable predictions.

Project Objective

The primary objective of this project is to develop a robust and well-optimized machine learning pipeline to predict house prices. The project emphasizes:
	•	Data Quality: Ensuring the data is clean, well-processed, and suitable for modeling.
	•	Model Performance: Comparing multiple models like XGBoost and TensorFlow-based MLP to choose the best-performing one.
	•	Interpretability: Visualizing feature importance and understanding model predictions through techniques like Feature Maps and Grad-CAM (upcoming).
	•	Production-Ready Code: Writing professional, well-structured, and well-documented code suitable for real-world applications.

Dataset

The dataset comes from the well-known Kaggle House Prices Competition.
	•	train.csv: Training data containing house features and their sale prices.
	•	test.csv: Test data containing house features without sale prices.

Both files should be placed in the /data directory as described in the project structure below.

Requirements

Before running the project, ensure you have a virtual environment set up and the necessary libraries installed.

1. Create and activate a virtual environment

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

How to Run

The entire pipeline — from data preprocessing to model training, evaluation, and visualization — is executed through the main.py file.

python main.py

Project Structure

House-Price-Prediction/
│
├── data/
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Test dataset
│
├── notebooks/
│   └── house_price_prediction.ipynb  # Exploratory data analysis and experimentation
│
├── src/
│   ├── data_preprocessing.py      # Cleans and preprocesses the data
│   ├── data_updater.py            # Updates and manages new data records
│   ├── model.py                   # Builds and trains machine learning models
│   ├── utils.py                   # Utility functions for model evaluation and visualization
│
├── venv/                          # Virtual environment (created separately)
│
├── .gitignore                     # Git ignore file
├── main.py                        # Main script to run the entire pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

Key Features
	•	Complete Machine Learning Pipeline: From data preprocessing to model evaluation and visualization.
	•	Multiple Model Comparison: Implementation of XGBoost and TensorFlow MLP for performance benchmarking.
	•	Hyperparameter Optimization: RandomizedSearchCV for efficient hyperparameter tuning.
	•	Cross-Validation: KFold cross-validation for robust model evaluation.
	•	Comprehensive Metrics: MSE, MAE, RMSE, and R² Score for well-rounded model assessment.
	•	Feature Importance Visualization: Graphical representation of feature impact on predictions.
	•	Upcoming Enhancements:
	•	Feature Maps Visualization for CNN interpretability.
	•	Grad-CAM Implementation for visualizing which areas of input data influence model predictions.

Next Steps
	•	Implement CNN feature map visualization.
	•	Add Grad-CAM to improve interpretability of CNN predictions.
	•	Further optimize the CNN architecture and hyperparameters for even better performance.