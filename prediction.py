import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils import process_dataframe, scaler_function, get_X_y  # Reuse existing functions
from gan_model.wgan_gp import Generator  # Load the trained generator

# Load trained generator model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function for rolling predictions
def rolling_predict(generator, df, X_scaler, y_scaler, n_steps_in, n_steps_out=1):
    """ Rolling prediction for 5 future days using a model trained for 1-step ahead predictions. """

    predicted_prices = []
    actual_prices = []
    dates = []
    
    for i in range(len(df) - n_steps_in - n_steps_out):
        X_input = df.iloc[i : i + n_steps_in].values  # Last n_steps_in data
        X_scaled = X_scaler.transform(X_input)
        X_scaled = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])  # Reshape for model
        
        predicted_steps = []
        current_input = X_scaled.copy()  # Copy initial input
        
        for step in range(n_steps_out):  # Predict next 5 days
            y_pred_scaled = generator.predict(current_input)  # Predict 1 day
            y_pred = y_scaler.inverse_transform(y_pred_scaled)  # Rescale
            
            predicted_steps.append(y_pred[0][0])  # Store predicted price
            
            # Update input with predicted value (rolling effect)
            current_input = np.roll(current_input, -1, axis=1)  # Shift left
            current_input[0, -1, 0] = y_pred_scaled[0][0]  # Replace last column with new prediction
        
        # Save actual and predicted values
        y_actual = df.iloc[i + n_steps_in : i + n_steps_in + n_steps_out]["Close"].values
      #   predicted_prices.append(predicted_steps)
      #   actual_prices.append(y_actual.tolist())
        actual_prices.append(y_actual[0])  # Store only the first actual price
        predicted_prices.append(predicted_steps[0])  
        dates.append(df.index[i + n_steps_in])

    return dates, actual_prices, predicted_prices

# Streamlit UI
st.title("📈 Stock Price Prediction (WGAN)")
st.sidebar.header("Select Stock & Date Range")

# User inputs
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

if st.sidebar.button("Predict"):
    # Fetch stock data
    df = yf.download(ticker, start=start_date - timedelta(days=100), end=end_date)
    df.reset_index(inplace=True)

    # Process dataframe
    df_processed = process_dataframe(df)
    
    # Load scalers and trained model
    generator_model = load_model("gen_model_14_1_199_indi.keras")  # Update with actual model path
    X_scaler, y_scaler = scaler_function(df_processed.iloc[:-5], df_processed[["Close"]].iloc[:-5])  # Fit scaler on past data

    # Perform rolling predictions
    dates, actual, predicted = rolling_predict(generator_model, df_processed, X_scaler, y_scaler, 14, 1)

    # Convert results to DataFrame
    results_df = pd.DataFrame({"Date": dates, "Actual": actual, "Predicted": predicted})
    results_df["Date"] = pd.to_datetime(results_df["Date"])
    results_df.set_index("Date", inplace=True)

    # Plot actual vs predicted prices
    st.subheader(f"Actual vs Predicted Stock Prices for {ticker}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df["Actual"], label="Actual Price", color="blue")
    ax.plot(results_df.index, results_df["Predicted"], label="Predicted Price", color="red", linestyle="dashed")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"{ticker} Stock Price Prediction")
    st.pyplot(fig)

    # Display predictions
    st.subheader("Predicted Prices (Next 5 Days)")
    st.write(results_df.tail(5))

# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# from utils import process_dataframe, scaler_function, get_X_y  # Reuse existing functions
# from gan_model.wgan_gp import Generator  # Load the trained generator

# # Load trained generator model
# @st.cache_resource
# def load_model(model_path):
#     return tf.keras.models.load_model(model_path)

# # Function for direct 5-day ahead predictions
# def predict_next_5_days(generator, df, X_scaler, y_scaler, n_steps_in, n_steps_out=5):
#     """ Predicts next 5 days based on the last available `n_steps_in` data. """
    
#     # Get the last `n_steps_in` days from the dataset
#     X_input = df.iloc[-n_steps_in:].values  
#     X_scaled = X_scaler.transform(X_input)
#     X_scaled = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])  # Reshape for model
    
#     # Predict next 5 days
#     y_pred_scaled = generator.predict(X_scaled)  
#     y_pred = y_scaler.inverse_transform(y_pred_scaled)  # Rescale predictions to original price range

#     # Generate future dates
#     last_date = df.index[-1]
#     future_dates = [last_date + timedelta(days=i) for i in range(1, n_steps_out + 1)]
    
#     return future_dates, y_pred.flatten()

# # Streamlit UI
# st.title("📈 Stock Price Prediction (WGAN)")
# st.sidebar.header("Select Stock & Date Range")

# # User inputs
# ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
# start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
# end_date = st.sidebar.date_input("End Date", datetime.today())

# if st.sidebar.button("Predict"):
#     # Fetch stock data
#     df = yf.download(ticker, start=start_date - timedelta(days=100), end=end_date)
#     df.reset_index(inplace=True)

#     # Process dataframe
#     df_processed = process_dataframe(df)
    
#     # Load scalers and trained model
#     generator_model = load_model("gen_model_14_1_199_indi.keras")  # Update with actual model path
#     X_scaler, y_scaler = scaler_function(df_processed.iloc[:-5], df_processed[["Close"]].iloc[:-5])  # Fit scaler on past data

#     # Perform direct next 5-day prediction
#     future_dates, predicted_prices = predict_next_5_days(generator_model, df_processed, X_scaler, y_scaler, 14, 5)

#     # Convert results to DataFrame
#     results_df = pd.DataFrame({"Date": future_dates, "Predicted": predicted_prices})
#     results_df["Date"] = pd.to_datetime(results_df["Date"])
#     results_df.set_index("Date", inplace=True)

#     # Plot predicted prices
#     st.subheader(f"Predicted Stock Prices for Next 5 Days ({ticker})")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(results_df.index, results_df["Predicted"], marker="o", color="red", linestyle="dashed", label="Predicted Price")
#     ax.legend()
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Stock Price")
#     ax.set_title(f"{ticker} Next 5-Day Price Forecast")
#     st.pyplot(fig)

#     # Display predictions
#     st.subheader("Predicted Prices (Next 5 Days)")
#     st.write(results_df)
