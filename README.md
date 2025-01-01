# Machine Learning Project for E-commerce Sales Prediction

This is a machine learning project designed to predict e-commerce sales using historical data. The goal is to develop a model that accurately forecasts future sales based on various factors such as price, units sold, and date.

## Technologies Used
- Python 3.x
- pandas
- NumPy
- scikit-learn
- matplotlib

## Setup Instructions

1. **Clone this repository:**
    ```bash
    git clone https://github.com/HuzaifaXool/Ecommerce-Prediction.git
    ```

2. **Navigate to the project folder:**
    ```bash
    cd Ecommerce-Prediction
    ```

3. **Create a virtual environment:**
    ```bash
    conda create --name venv
    ```

4. **Activate the environment (as this project was developed using Linux):**
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```

5. **Install required libraries:**
    ```bash
    python3 setup.py install
    ```

6. **Running the main program**:
    After completing the setup, you can execute the main program and generate sales predictions by running the following command:
    ```bash
    python main.py
    ```

    - `main.py`: This script contains the code to load the data, preprocess it, and train the machine learning models.

## Description of Processes and Files

### Data Processing
- The dataset is loaded and cleaned by `dataloader.py`.
- It handles missing values, outliers, and summarization of key statistics.

### Feature Engineering
- Identified and removed unnecessary columns such as `Date`.
- Encoded categorical variables like `Customer_Segment`.
- Scaled numerical features like `Price` and `Marketing_Spend`.

### Model Development
- Various machine learning models have been explored (e.g., Linear Regression, Random Forest).
- The models are trained using the preprocessed data, and performance is evaluated based on sales prediction accuracy.

### Future Work
- Experimenting with hyperparameter tuning.
- Extending the dataset to improve model performance.
- Optimizing the feature engineering process.


``` bash
E-commerce-Sales/House-price-prediction
├── build
│   ├── bdist.linux-x86_64
│   └── lib
│       ├── src
│       │   ├── data
│       │   │   ├── data_loader.py         # Loads raw data into the environment
│       │   │   ├── data_preprocessing.py  # Handles preprocessing tasks like handling nulls
│       │   │   ├── feature_engineering.py  # Implements feature engineering methods
│       │   │   └── __init__.py
│       │   ├── __init__.py
│       │   ├── main.py                    # Main entry point for executing the project
│       │   ├── models
│       │   │   ├── __init__.py
│       │   │   ├── model_evaluation.py    # Code for evaluating model performance
│       │   │   └── model.py                # Defines machine learning models
│       │   └── training
│       │       ├── callback.py            # Custom training callback for model training
│       │       ├── __init__.py
│       │       └── train_pipeline.py      # Implements the training pipeline logic
│       └── tests
│           ├── __init__.py
│           ├── test_data.py              # Test scripts for data processing functions
│           └── test_model.py             # Test scripts for model evaluation and predictions
├── data
│   ├── archive.zip                       # Dataset archive file
│   ├── Ecommerce_Sales_Prediction_Dataset.csv # Raw e-commerce sales dataset
│   └── housing_price_dataset.csv         # Raw housing price dataset
├── dist
│   ├── ml_project-1-py3.10.egg           # Project package file for Python 3.10
│   ├── ml_project-1-py3.12.egg           # Project package file for Python 3.12
│   └── ml_project-1.tar.gz               # Compressed archive for the project package
├── environment.txt                       # Lists the required environment dependencies
├── feature_engineered_data.csv           # Dataset after feature engineering steps
├── joblib
├── ml_project.egg-info                  # Metadata for the project package
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── pkl_files
│   ├── linear_model.pkl                  # Saved model file for the linear model
│   ├── onehot_encoder.pkl               # Saved encoder for categorical features
│   └── scaler.pkl                       # Saved feature scaling transformer
├── processed_data
│   ├── cleaned_data.csv                 # Cleaned dataset after preprocessing
│   ├── feature_engineered_data
│   │   └── feature_engineered_data.csv  # Data after feature engineering
│   ├── test.csv                          # Processed test dataset
│   └── train.csv                         # Processed train dataset
├── README.md                            # Project description and instructions
├── requirement.txt                      # Contains the required Python libraries
├── setup.py                             # Installation script for the project
├── src
│   ├── data
│   │   ├── data_loader.py               # Loads dataset into the Python environment
│   │   ├── feature_engineering.py       # Defines feature engineering methods
│   │   └── __init__.py
│   ├── __init__.py
│   ├── main.py                          # Main script to run the project pipeline
│   ├── models
│   │   ├── __init__.py
│   │   ├── model_evaluation.py         # Evaluates model performance
│   │   └── model.py                     # Defines various machine learning models
│   ├── training
│   │   ├── callback.py                  # Custom callback during training
│   │   ├── __init__.py
│   │   ├── model_traning.ipynb         # Jupyter notebook for model training
│   │   └── train_pipeline.py            # Code that defines the model training pipeline
│   └── vizualization
│       ├── =4.2.0
│       ├── data_visualization.py        # Code for visualizing the data
│       ├── Untitled.ipynb
│       └── Visualization.ipynb          # Jupyter notebook with various visualizations
├── tests
│   ├── __init__.py
│   ├── test_data.py                     # Unit tests for the data processing code
│   └── test_model.py                    # Unit tests for the model code
└── visualization_files
    ├── Average_Discounts_by_Product_Category.png  # Visualization of discount distribution
    ├── Comparison_of_Unit_Price_Across_Product_Categories.png  # Unit price comparison
    ├── Correlation_of_Numerical_Values.png # Correlation heatmap for numerical variables
    ├── Count_of_Product_Categories_Sold.png # Category-wise sales count plot
    ├── Distribution_of_Discount.png       # Distribution of product discounts
    ├── Distribution_of_Units_Sold.png     # Distribution of units sold
    ├── Marketing_Spend_Distribution.png  # Distribution of marketing spend
    ├── Monthly_Sales_from_2023_to_2025.png # Monthly sales data trend visualization
    ├── Price_Distribution_by_Product_Category_and_Customer_Segment.png  # Price distribution
    ├── Product_Category_ROI_Analysis.png  # Return on investment for product categories
    ├── Revenue_Distribution_by_Product_Category_and_Customer_Segment.png # Revenue insights
    ├── Total_Sales_of_Products_by_Category.png  # Total sales for different product categories
    └── Yearly_Sales_by_Product_Category.png   # Yearly sales distribution by category
```