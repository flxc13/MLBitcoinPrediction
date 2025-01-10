```markdown
# BTC/USDT Price Movement Prediction

This repository contains code for predicting BTC/USDT price movement using historical OHLCV (Open, High, Low, Close, Volume) data fetched from Binance using the `ccxt` library. The prediction model is built using a Support Vector Machine (SVM) classifier with hyperparameter tuning and various technical indicators.

---

## **Features**

### **Data Collection**
- Retrieves historical OHLCV data for BTC/USDT from Binance using the `ccxt` library.
- Supports data fetching in a loop to handle API limitations (1000 data points per request).
- Data preprocessing includes:
  - Timestamp conversion to datetime.
  - Calculation of percentage returns based on closing prices.
  - Binary classification label based on a return threshold (e.g., 0.05%).

### **Feature Engineering**
- Computes various technical indicators for financial data analysis:
  - Open-Close difference.
  - High-Low difference.
  - Simple Moving Average (SMA).
  - Exponential Moving Average (EMA).
  - Relative Strength Index (RSI).
  - Moving Average Convergence Divergence (MACD).
  - Bollinger Bands.
- Encapsulates feature computation in a reusable `FeaturesComputation` class.

### **Data Processing**
- Data standardization using `StandardScaler`.
- Train-test split (chronological order maintained for time-series data) using `train_test_split`.

### **Model Building**
- SVM classifier with grid search for hyperparameter tuning.
- Cross-validation using `TimeSeriesSplit`.
- Evaluation metrics include:
  - Accuracy.
  - Classification report.
  - Confusion matrix.
  - ROC curve and AUC score.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/btc-usdt-prediction.git
   cd btc-usdt-prediction
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **1. Data Fetching**
The historical BTC/USDT data is fetched using the `ccxt` library with the `fetch_ohlcv` function. This step ensures:
- Data is fetched from January 1, 2018, with a 6-hour timeframe.
- Data is preprocessed and stored in a pandas DataFrame.

Example:
```python
symbol = 'BTC/USDT'
timeframe = '6h'
since = ccxt.binance().parse8601('2018-01-01T00:00:00Z')
stock_data = fetch_ohlcv(symbol, timeframe, since)
```

### **2. Feature Engineering**
Technical indicators are computed using the `FeaturesComputation` class. Example:
```python
features_df = pd.DataFrame(FeaturesComputation(stock_data).compute_features())
features_df.index = stock_data.index
features_df.dropna(inplace=True)
X = features_df
y = stock_data['Label'].loc[X.index]
```

### **3. Data Processing**
The `DataProcess` class standardizes the feature matrix and splits the data into training and testing sets:
```python
data_process = DataProcess(X, y, testsize=0.2)
X_train, X_test, y_train, y_test = data_process.X_train, data_process.X_test, data_process.y_train, data_process.y_test
```

### **4. Model Building and Evaluation**
The `SVM_Classifier` function builds and evaluates the SVM model:
```python
SVM_Classifier(X_train, y_train, X_test, y_test)
```

---

## **Results**
- **Training Accuracy**: 64.85%  
- **Test Accuracy**: 67.32%  
- **AUC (Area Under ROC Curve)**: 0.728  

The model evaluation includes:
1. Confusion matrix.
2. Classification report.
3. ROC curve visualization.

---

## **File Structure**
```
btc-usdt-prediction/
├── main code.ipynb      # Jupyter Notebook containing the complete code
├── requirements.txt     # List of required Python libraries
├── README.md            # Project documentation
```

---

## **Dependencies**
The following Python libraries are required:
- `ccxt`
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`

Install them using:
```bash
pip install -r requirements.txt
```

---

## **Acknowledgments**
- **Binance API**: Historical crypto data fetched using the Binance API via the `ccxt` library.
- **Scikit-learn**: Machine learning library used for data preprocessing and modeling.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE).

---

## **Contributing**
Feel free to contribute by submitting issues or pull requests. Contributions that improve the model or add new features are welcome!
```
