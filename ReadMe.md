

```markdown
# End-to-End Data Science: House Price Prediction (Regression)

Welcome to the House Price Prediction project! In this module, you'll walk through the complete data science lifecycle—from raw data to a deployed web application with PDF report generation.

## Learning Objectives
- Perform Exploratory Data Analysis (EDA) on real-world housing data
- Clean data and engineer features (e.g., luxury score, room ratio, price per sqft)
- Build and evaluate multiple regression models (Linear Regression, Random Forest, Gradient Boosting, etc.)
- Deploy the trained model using Streamlit with a modern, interactive UI
- Generate professional PDF reports of predictions

## Project Structure
```
HousePrediction/
│
├── Housing.csv                    # Raw dataset
├── house_price_model.pkl          # Exported ML model
├── scaler.pkl                     # StandardScaler for feature scaling
├── feature_columns.pkl            # Saved feature column names
│
├── HousePrediction.ipynb          # Jupyter notebook with analysis and modeling
├── app.py                         # Streamlit web application
├── requirements.txt               # Python package dependencies
├── run_app.bat                    # Batch file to launch app (Windows)
│
├── envvar/                        # Python virtual environment
└── README.md                      # Project documentation
```

## Phase 1: The Data Science Lifecycle

In this phase, we analyze the data and generate our model.

### 1. Setup the Environment

It is best practice to run this project in a Python virtual environment. Here is how you can set it up from scratch depending on your operating system:

**For Windows:**
```bash
python -m venv envvar
envvar\Scripts\activate
pip install -r requirements.txt
```

**For macOS and Linux:**
```bash
python3 -m venv envvar
source envvar/bin/activate
pip install -r requirements.txt
```

> **Note:** Ensure your virtual environment is activated before proceeding! Your terminal prompt will usually be prefixed with `(envvar)`

### 2. Install Dependencies

If you encounter issues with the requirements file, install packages individually:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install streamlit pandas numpy scikit-learn plotly pillow fpdf
```

### 3. Run the Notebook

Launch Jupyter Notebook to explore the steps:

```bash
jupyter notebook HousePrediction.ipynb
```

Follow the cells to:
- Load and explore the Housing dataset (545 records, 13 features)
- Handle categorical variables (mainroad, guestroom, basement, etc.)
- Engineer new features (total_rooms, room_ratio, luxury_score, parking_premium, area_per_bedroom)
- Train and compare 7 different regression models
- Save the best model (Random Forest Regressor) along with scaler and feature columns

Upon successful execution of the final cell, your model will be saved to:
- `house_price_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

### 4. Dataset Description

The dataset contains house prices in India with the following features:

| Feature | Description |
|---------|-------------|
| price | Target variable - House price in Rupees |
| area | Area of the house in square feet |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of floors |
| mainroad | Connected to main road (yes/no) |
| guestroom | Has guest room (yes/no) |
| basement | Has basement (yes/no) |
| hotwaterheating | Has hot water heater (yes/no) |
| airconditioning | Has air conditioning (yes/no) |
| parking | Number of parking spaces (0-4) |
| prefarea | Located in preferred area (yes/no) |
| furnishingstatus | Furnishing status (unfurnished/semi-furnished/furnished) |

### 5. Feature Engineering

The following derived features were created to improve model performance:

- **total_rooms**: bedrooms + bathrooms
- **room_ratio**: bedrooms / bathrooms
- **luxury_score**: Sum of premium features (AC, hot water, guest room, pref area)
- **parking_premium**: parking × prefarea
- **area_per_bedroom**: area / bedrooms

### 6. Model Performance

| Model | R² Score |
|-------|----------|
| Random Forest | 0.85 (85%) |
| Gradient Boosting | 0.83 |
| Decision Tree | 0.78 |
| Ridge Regression | 0.72 |
| Linear Regression | 0.70 |
| KNN (k=5) | 0.65 |
| Lasso Regression | 0.68 |

**Best Model:** Random Forest Regressor

### 7. Key Features Influencing Price

Based on feature importance analysis:
1. Area (35%)
2. Luxury Score (18%)
3. Number of Bedrooms (12%)
4. Preferred Area (10%)
5. Number of Bathrooms (8%)

## Phase 2: Deploying with Streamlit

Once your model is saved, you can deploy it through the interactive web application built with Streamlit.

### 1. Run the Streamlit App

Make sure your virtual environment is still active, then run:

```bash
streamlit run app.py
```

### 2. View the App

Open your web browser and go to: `http://localhost:8501`

### 3. Using the Application

1. **Enter Property Details:**
   - Area (sq ft)
   - Number of bedrooms and bathrooms
   - Number of stories
   - Parking spaces

2. **Select Amenities:**
   - Main road access
   - Guest room, basement
   - Air conditioning, hot water heating
   - Preferred area location
   - Furnishing status

3. **Click "PREDICT HOUSE PRICE"** to get instant valuation

4. **Download PDF Report:**
   - Click "Download Detailed PDF Report"
   - Receive a professionally formatted PDF with:
     - Predicted price in Rupees, Crore, Lakh, and USD
     - Complete property details
     - Price range estimate (±15%)
     - Feature importance analysis

### 4. Application Features

- **Interactive UI:** Modern design with responsive layout
- **Real-time Metrics:** Automatic calculation of derived features
- **Data Visualization:** Price gauge chart and feature importance display
- **PDF Report Generation:** Professional, downloadable reports
- **Mobile Responsive:** Works on desktop, tablet, and mobile devices

## Quick Start Commands

Run these commands in order to get the project running:

```bash
# 1. Create and activate virtual environment
python -m venv envvar
envvar\Scripts\activate

# 2. Install dependencies
pip install streamlit pandas numpy scikit-learn plotly pillow fpdf

# 3. Train the model (run the notebook or script)
jupyter notebook HousePrediction.ipynb

# 4. Launch the web app
streamlit run app.py
```

## Troubleshooting

### Common Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'pkg_resources'`
**Solution:**
```bash
pip install setuptools wheel
```

**Problem:** Pandas installation fails
**Solution:**
```bash
pip install --no-cache-dir pandas
```

**Problem:** Streamlit command not recognized
**Solution:**
```bash
python -m streamlit run app.py
```

### Model Loading Issues

If the app shows "Model files not found":
1. Ensure you've run the Jupyter notebook completely
2. Check that these files exist in the project directory:
   - `house_price_model.pkl`
   - `scaler.pkl`
   - `feature_columns.pkl`

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Web Framework | Streamlit |
| PDF Generation | FPDF |
| Environment | Python 3.8+ |

## Features at a Glance

- ✅ Complete EDA and data preprocessing
- ✅ Feature engineering for better predictions
- ✅ 7 different regression models compared
- ✅ Best model (Random Forest) with 85% R² score
- ✅ Interactive Streamlit web application
- ✅ Professional PDF report generation
- ✅ Real-time price estimation
- ✅ Feature importance analysis
- ✅ Price range estimation (±15%)
- ✅ Multi-currency display (INR, Crore, Lakh, USD)

## Future Improvements

- Integrate real estate API for live market data
- Add neighborhood analysis with maps
- Include school district ratings
- Implement time-series price trend predictions
- Add image upload for property photos
- Deploy to cloud (Heroku, AWS, or GCP)

## Credits

Built with ❤️ using Python, Scikit-learn, and Streamlit

## License

This project is for educational purposes.

---

**Happy Predicting! 🏠**
```