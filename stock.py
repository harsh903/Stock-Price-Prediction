import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX




def main():
    # Set page title and favicon
    st.set_page_config(page_title='Stock Price Prediction', page_icon=':money_with_wings:')

    # Set app title and header
    st.title('Stock Price Prediction')
    st.markdown('Use this app to predict stock prices for different companies.')

    # Define the sidebar with input options
    with st.sidebar:
        st.subheader('Input Options')
        # Get the company name from the user
        company = st.selectbox('Select the company', ['Microsoft (MSFT)', 'Apple (AAPL)', 'Google (GOOGL)', 'Amazon (AMZN)', 'Facebook (FB)'])

        # Get the date range from the user
        start_date = st.date_input('Start date', value=pd.to_datetime('2019-01-01'), min_value=pd.to_datetime('2010-01-01'), max_value=pd.to_datetime('2021-12-31'))
        end_date = st.date_input('End date', value=pd.to_datetime('2019-12-31'), min_value=pd.to_datetime('2010-01-01'), max_value=pd.to_datetime('2021-12-31'))
        if end_date < start_date:
            st.warning('End date must be after start date.')
            return


        # Add a predict button to trigger the stock price prediction
        predict = st.button('Predict')

    # Download the stock prices from Yahoo Finance and train the ARIMA model
    if predict:
        if company == 'Microsoft (MSFT)':
            ticker = 'MSFT'
        elif company == 'Apple (AAPL)':
            ticker = 'AAPL'
        elif company == 'Google (GOOGL)':
            ticker = 'GOOGL'
        elif company == 'Amazon (AMZN)':
            ticker = 'AMZN'
        elif company == 'Facebook (FB)':
            ticker = 'FB'

        # Download stock data from Yahoo Finance
        stock_data = yf.download(ticker, start='2005-01-01', end='2022-12-31')

        train_data = stock_data['Open'][:'2022']
        test_data = stock_data['Open'][start_date:end_date]  # Changed the test data date range to match the user input



        # Train an ARIMA model on the stock data
        model = SARIMAX(train_data, order=(1, 1, 2), seasonal_order=(1, 1, 1, 12)).fit()

        # Predict the stock prices for the given date range
        start_date1 = test_data.index.min()
        end_date1 = test_data.index.max()
        predictions = model.predict(start=start_date1, end=end_date1, dynamic=True)

        # Plot the predicted prices
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(test_data)
        ax.plot(predictions, color='r')
        ax.set_title(f'{company} Stock Prices (Open) from {start_date} to {end_date}')
        ax.legend(['Actual', 'Predicted'])
        st.pyplot(fig)
        st.write('This graph shows the predicted and actual stock prices (open) for ' + company +
         ' from ' + str(start_date) + ' to ' + str(end_date) + '. The blue line represents the actual stock prices, while the red line represents the predicted stock prices using an SARIMAX model trained on the data up to ' + str(pd.to_datetime('2022-01-01')) + '. The predictions should be taken as a guide and not as a guarantee of future stock prices.')
if __name__ == '__main__':
    main()
