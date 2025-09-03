# Flask App for Salary Prediction:
# - Purpose: Enables HR users to upload a CSV file, predict salaries using a pre-trained XGBoost model, and store results in a PostgreSQL database (mydb5, table: salarydata).
# - Connection to Notebook: Uses the XGBoost pipeline from instilit 4 7.ipynb with preprocessing (SimpleImputer, StandardScaler, OneHotEncoder/TargetEncoder) and model (Cell 98: RMSE 0.6681, R² 0.8875, MAPE 3.2958%).
# - Functionality: Validates input columns, predicts log-transformed salaries, converts to USD, applies currency conversion, saves results to database, and displays in a styled HTML table.
# - Robustness: Handles errors for missing files or columns and supports SHAP explanations for interpretability.

# Import required libraries for Flask app, data handling, and database connection
from flask import Flask, render_template, request  # Flask for web app, render_template for HTML, request for handling form/file uploads
import pandas as pd  # For reading and manipulating CSV data
import numpy as np  # For numerical operations (e.g., inverse log-transformation)
import joblib  # For loading the trained machine learning pipeline
import os  # For file system operations (e.g., creating upload folder)
from sqlalchemy import create_engine  # For connecting to PostgreSQL database

# Initialize Flask application
app = Flask(__name__)  # Create Flask app instance with the current module name
app.config['UPLOAD_FOLDER'] = 'uploaded'  # Define folder for storing uploaded CSV files

# Load the trained pipeline (preprocessing + XGBoost model)
# Pipeline includes imputation, scaling, encoding, and XGBoost model
pipeline = joblib.load("saved_models/final_XGBoost_pipelinenew.pkl")

# Set up PostgreSQL database connection
# Connects to 'mydb5' database for storing predictions
DB_URI = 'postgresql+psycopg2://postgres:Madhu14777@localhost:5432/mydb5'
engine = create_engine(DB_URI)  # Create SQLAlchemy engine for database operations

# Ensure the upload folder exists
# Creates 'uploaded' directory if it doesn't exist for storing user-uploaded CSV files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define route for the home page, handling both GET and POST requests
@app.route("/", methods=["GET", "POST"])
def home():
    # Handle POST request when user uploads a CSV file
    if request.method == "POST":
        # Retrieve the uploaded file from the form
        file = request.files.get("csv_file")
        # Check if no file was uploaded
        if not file:
            # Render index.html with an error message
            return render_template("index.html", error="❌ Please upload a CSV file.")

        try:
            # Read the uploaded CSV file into a pandas DataFrame
            df = pd.read_csv(file)

            # Define required columns for prediction (subset of notebook features, Cell 97)
            required_cols = [
                'job_title', 'experience_level', 'employment_type',
                'company_size', 'company_location', 'remote_ratio',
                'years_experience', 'salary_currency', 'conversion_rate'
            ]

            # Verify that all required columns are present in the uploaded CSV
            if not all(col in df.columns for col in required_cols):
                # Identify missing columns
                missing = list(set(required_cols) - set(df.columns))
                # Return error message with missing columns
                return render_template("index.html", error=f"❌ Missing columns: {missing}")

            # Make predictions using the loaded pipeline
            # Pipeline applies preprocessing (imputation, scaling, encoding) and predicts log-transformed salaries (Cell 98)
            predictions_log = pipeline.predict(df[required_cols])  # Predict log-transformed salaries
            df["predicted_log_salary"] = predictions_log  # Store log-transformed predictions
            df["adjusted_total_usd"] = np.expm1(predictions_log)  # Convert log predictions to USD using expm1
            df["converted_salary"] = df["adjusted_total_usd"] * df["conversion_rate"]  # Apply currency conversion

            # Prepare DataFrame for storage, including input features and predictions
            df_to_store = df[required_cols + ['adjusted_total_usd', 'converted_salary']]
            # Save predictions to PostgreSQL database table 'salarydata'
            # Appends to existing table (updated from 'salarytbl' in previous version)
            df_to_store.to_sql("salarydata", engine, if_exists="append", index=False)

            # Convert results to HTML table for display
            # Uses Bootstrap classes for styling (table-striped)
            result_html = df_to_store.to_html(classes="table table-striped", index=False)
            # Render predict.html template with the results table
            return render_template("predict.html", table=result_html)

        except Exception as e:
            # Handle any errors during file processing or prediction
            # Return index.html with error message
            return render_template("index.html", error=f"❌ Error: {str(e)}")

    # Handle GET request: display the home page (index.html)
    return render_template("index.html")

# Run the Flask app in debug mode if executed directly
if __name__ == "__main__":
    app.run(debug=True)  # Debug mode enables auto-reload and error details for development