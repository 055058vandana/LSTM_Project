# **LSTM-Based Stock Price Prediction**

## **Contributors:**  
- **Shefali Pujara (055044):**
- **Vandana Jain (055058):**

---

## **Introduction & Objective**
Stock market prediction remains a challenging endeavor due to the inherent volatility and non-linearity of financial data. Traditional statistical models, like ARIMA, often struggle to capture the complex relationships and dependencies within time series data, leading to limited accuracy in forecasting future stock prices.  

This project addresses these limitations by leveraging the power of **Long Short-Term Memory (LSTM)** networks, a type of recurrent neural network specifically designed to handle sequential data. The objective is to develop an LSTM-based model capable of accurately predicting stock prices for **Bank of India** and the **NIFTY 50** index. The project encompasses data preprocessing, LSTM model development, rigorous training and evaluation, and a comparative analysis with the traditional ARIMA model to assess the LSTM's effectiveness in capturing stock market dynamics.

---

## **Data Collection**
Stock price data for Bank of India (BANKINDIA.NS) and the NIFTY 50 index (^NSEI) was obtained using the **`yfinance`** library, a popular Python package for downloading financial data from Yahoo Finance.  

The dataset retrieved covers a historical period from **September 17, 2007, to March 25, 2025** (or the most recent available data). It includes the following essential features for each trading day:
- **Open:** The price at which the stock opened for trading.
- **High:** The highest price reached during the trading day.
- **Low:** The lowest price reached during the trading day.
- **Close:** The price at which the stock closed for trading.
- **Volume:** The number of shares traded during the day.

---

## **Exploratory Data Analysis (EDA)**
### **Summary Statistics & Visualizations**
Before model development, a thorough exploratory data analysis was conducted to gain insights into the characteristics and patterns of the stock price data.

- **Descriptive Statistics:** Summary statistics such as mean, standard deviation, minimum, maximum, and quartiles were calculated for the stock prices to understand the distribution and variability of the data.
- **Line Charts:** Line charts were plotted to visualize the historical trends and patterns in the closing prices of both Bank of India and the NIFTY 50 index over time. This allowed us to identify periods of growth, decline, and volatility.
- **Correlation Heatmap:** A correlation heatmap was generated to analyze the relationships between different stock features (open, high, low, close, volume). This helped us understand the interdependence between these features and identify potential predictors for the model.

---

## **Feature Engineering**
To enhance the predictive power of the LSTM model, several feature engineering techniques were applied:

- **Lag Features:** Previous stock prices were included as features, creating lagged variables (e.g., Close price of the previous day, Close price of two days ago). These lagged features allow the model to learn from past price movements and identify potential trends.
- **Relative Strength Index (RSI):** The RSI, a momentum indicator, was calculated to assess the magnitude of recent price changes and identify overbought or oversold conditions. This technical indicator helps capture the stock's momentum and potential for reversals.
- **Moving Averages:** 10-day and 50-day moving averages were computed to smooth out price fluctuations and identify the underlying trend direction. Moving averages provide a clearer picture of the overall price movement and can act as support or resistance levels.

---

## **Data Preprocessing & Train-Test Split**
To prepare the data for model training, the following preprocessing steps were performed:

- **Missing Values Handling:** Any missing values in the dataset were addressed using appropriate methods, such as imputation or removal, ensuring data integrity.
- **Data Normalization:** The data was normalized using `MinMaxScaler` from the `sklearn.preprocessing` module. This scales the features to a range between 0 and 1, improving model convergence and stability during training.
- **Train-Test Split:** The dataset was carefully divided into training, validation, and test sets to ensure a robust evaluation of the model's performance. A clear distinction was maintained between the data used for model learning (training set), hyperparameter tuning (validation set), and final performance assessment (test set).

---

## **LSTM Model Training**
An **LSTM-based neural network** was constructed for stock price prediction. The model architecture incorporated:

- **LSTM Layers:** One or more LSTM layers were used to capture the temporal dependencies and long-term patterns within the stock price data.
- **Dropout Layers:** Dropout layers were added to prevent overfitting and improve the model's generalization ability.
- **Dense Layers:** Dense layers were used for the output layer and potentially intermediate layers to learn complex relationships between the features and the target variable (stock price).
- **Adam Optimizer:** The Adam optimizer was employed to update the model's weights during training, facilitating efficient convergence.
- **Mean Squared Error (MSE):** The Mean Squared Error was used as the loss function, quantifying the difference between the predicted and actual stock prices.
- **Training Process:** The model was trained over multiple epochs, and the training and validation losses were closely monitored to assess the model's learning progress and prevent overfitting.

---

## **Model Performance Evaluation**
The performance of the LSTM model was rigorously evaluated using the following error metrics:

- **Root Mean Squared Error (RMSE):** Measures the average magnitude of the prediction errors.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between the predicted and actual values.
- **Mean Absolute Percentage Error (MAPE):** Measures the average percentage error in the predictions.
- **Visualization:** Line charts were used to visualize the actual versus predicted stock prices, providing a visual assessment of the model's accuracy and ability to capture trends.


---

## **Comparison with ARIMA**
To benchmark the performance of the LSTM model, a traditional **ARIMA (Autoregressive Integrated Moving Average)** model was implemented as a baseline.

- **ARIMA Model:** An appropriate ARIMA model was selected based on the characteristics of the time series data.
- **Performance Comparison:** The RMSE and MAPE values of both the LSTM and ARIMA models were compared to determine which model provided better predictive accuracy.

---

## **Final Predictions & Visualizations**

### **Predicted vs. Actual Stock Prices**
To visually assess the predictive accuracy of the LSTM model, line charts were generated, comparing the predicted stock prices against the actual stock prices for both the validation and test sets. These visualizations provide a clear depiction of how well the model captures the overall trends and patterns in the stock price movements.

### **Residual Plot**
A residual plot was created by plotting the difference between the predicted and actual stock prices. This plot helps analyze the model's biases and identify any systematic patterns in the prediction errors. Ideally, the residuals should be randomly distributed around zero, indicating that the model is unbiased.

---

## **Observations**

### **Dataset Preparation**
- **Data normalization:** Applying MinMaxScaler to normalize the input data significantly improved the convergence speed and stability of the LSTM model during training. This preprocessing step ensures that all features are on a similar scale, preventing certain features from dominating the learning process and leading to faster and more reliable model training.

### **Model Performance**
- **Trend capture:** The LSTM model demonstrated an impressive ability to capture the overall trends and patterns in the stock price data. This indicates its effectiveness in learning the temporal dependencies and long-term relationships within the time series.
- **Prediction accuracy:** The model achieved relatively low RMSE and MAE values on the validation and test sets, signifying good prediction accuracy. These metrics provide quantitative evidence of the model's ability to generate accurate forecasts.

### **Comparison with ARIMA**
- **Non-linear dependencies:** The LSTM model outperformed the ARIMA model in capturing non-linear dependencies within the stock price data. This highlights the advantage of LSTMs in handling complex, non-linear relationships that are often present in financial time series.
- **Short-term trends:** While the LSTM model excelled in capturing long-term trends and patterns, the ARIMA model appeared to be better suited for predicting short-term fluctuations. This observation suggests that combining both models could potentially lead to a more comprehensive forecasting approach.

### **Prediction Accuracy**
- **General trends:** The LSTM model accurately captured the general trends and direction of stock price movements. This makes it a valuable tool for understanding the overall market dynamics and potential future directions.
- **Extreme fluctuations:** While the model performed well in stable market conditions, it exhibited slight deviations during periods of extreme price fluctuations. This indicates that external factors and market sentiment, which are not explicitly captured in the model's input features, can significantly impact stock prices during volatile periods.

---

## **Managerial Insights**

### **Data-Driven Decision Making**
- **Predictive insights:** The LSTM model provides valuable predictive insights into potential future stock price movements. This information can be leveraged by investors and traders to make informed investment decisions and develop data-driven strategies.
- **Buy/sell strategies:** The model's predictions can help optimize buy and sell strategies by identifying potential entry and exit points based on anticipated price trends.

### **Risk Management**
- **Sudden price changes:** The model's limitations in predicting extreme fluctuations highlight the importance of considering external factors and market sentiment. Incorporating macroeconomic indicators and sentiment analysis could potentially improve the model's robustness and ability to anticipate sudden price changes.
- **Risk mitigation:** While the model's predictions are valuable, it is crucial to acknowledge its limitations and use it as a tool to support decision-making, not as the sole basis for investment decisions. Diversification and other risk management strategies should be employed to mitigate potential losses.

### **Portfolio Optimization**
- **Balance and diversification:** LSTM forecasting can assist investors in balancing and diversifying their portfolios by providing insights into the expected performance of different stocks and assets. This helps create a portfolio that is aligned with the investor's risk tolerance and financial goals.

### **Automation in Financial Analysis**
- **Efficiency and scalability:** The use of AI-powered models like LSTM significantly reduces the manual effort involved in stock analysis and trading. This automation enables faster and more scalable analysis, allowing investors to process large amounts of data and identify potential opportunities more efficiently.

---

## **Future Enhancements**

### **Hybrid models**
- **Sentiment analysis and macroeconomic indicators:** Integrating sentiment analysis and macroeconomic indicators as additional input features could enhance the model's predictive power, particularly during volatile market conditions. Sentiment analysis captures the overall market mood and investor sentiment, while macroeconomic indicators provide insights into broader economic trends that can influence stock prices.
- **Combining with other models:** Exploring hybrid models that combine LSTM with other forecasting techniques, such as ARIMA or GARCH, could further improve accuracy and capture a wider range of market dynamics.

### **Hyperparameter tuning**
- **Optimization and adaptability:** Fine-tuning the hyperparameters of the LSTM model through techniques like grid search or Bayesian optimization can further optimize its performance and adaptability to different market conditions. This ensures that the model is tailored to the specific characteristics of the data and can generalize well to unseen data.

---

## **Conclusion**

This study demonstrates the effectiveness of LSTM networks for stock price prediction, particularly in stable market conditions. The model effectively captures general trends and patterns, providing valuable insights for data-driven decision-making in financial markets.

However, the model's limitations in predicting extreme fluctuations highlight the need for incorporating external factors and market sentiment to improve accuracy. Future enhancements, such as hybrid modeling with sentiment analysis and macroeconomic indicators, as well as hyperparameter tuning, hold the potential to further refine the model's performance and adaptability to different market conditions.

This project contributes to the growing body of research on AI-driven stock market analysis and provides valuable insights for investors and traders seeking to leverage advanced technologies for informed decision-making and portfolio management.
