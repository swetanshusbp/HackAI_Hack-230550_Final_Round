# Stock Price Prediction Using LSTM

## Overview

This project is designed to predict stock prices using a Long Short-Term Memory (LSTM) neural network. The LSTM model is implemented using Keras with TensorFlow as the backend. The project also includes data visualization tools for stock price data.

## Model

The LSTM model included in this project is tailored for time series prediction, particularly for predicting stock prices. It's structured to understand and learn patterns from 60 days of stock price history to predict the following day's closing price.

## Thought Process

- Data acquisition is done via the `yfinance` library, which provides a convenient way to download historical stock data.
- Data visualization is managed with `matplotlib` and `seaborn`, which are used to plot the stock price trends and moving averages.
- The LSTM model is designed with a sequence length of 60 days, considering the daily closing price as the primary feature.
- The model consists of LSTM layers followed by Dense layers, utilizing mean squared error as the loss function and the Adam optimizer.


## Improvements

- Expand the feature set to include more indicators, such as volume, open-high-low-close (OHLC) data, and technical indicators.
- Experiment with different model architectures and hyperparameters.
- Incorporate a validation set for better evaluation of the model's performance.

## Disclaimer

The predictions made by the LSTM model are based on historical data and are not guaranteed to reflect future performance. This tool should not be used as the sole basis for any investment decisions. The creators of this project bear no responsibility for any financial losses incurred as a result of using this model.



