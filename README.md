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

