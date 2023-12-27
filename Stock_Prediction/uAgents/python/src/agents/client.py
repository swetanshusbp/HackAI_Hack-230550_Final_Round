#imports for uAgents
from uagents import Agent, Bureau, Context, Model

#imports for prediction
from keras.models import load_model
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

#Model and prediction

# Function to prepare data for LSTM model
def prepare_lstm_data(data, feature, training_data_ratio=0.95):
    dataset = data.filter([feature]).values
    training_data_len = int(np.ceil(len(dataset) * training_data_ratio))

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Creating training data
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler, training_data_len

# Function to make predictions using LSTM model
def predict_with_lstm(model, data, feature, training_data_len):
    dataset = data.filter([feature]).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

# Modified function to plot predictions and return base64 string
def plot_predictions_base64(train, valid):
    plt.figure(figsize=(16, 6))
    plt.title('Model Predictions vs Actual Prices')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Actual', 'Predictions'], loc='lower right')

    # Display the plot
    plt.show()

    # Save plot to a BytesIO buffer for base64 encoding
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)  # Move to the beginning of the buffer
    plt.close()  # Close the figure

    # Encode the buffer to base64
    base64_encoded = base64.b64encode(buffer.getvalue()).decode()

    return base64_encoded




# Check if the data is empty

    
def predict_main(stock_symbol, start_date):
    # Load the previously trained model
    model = load_model('lstm_model.keras')


    # Download data for the entered stock
    stock_data = yf.download(stock_symbol, start=start_date, end=datetime.now())
    if stock_data.empty:
        print(f"No data found for {stock_symbol}. Please check the stock symbol and try again.")
    else:
        # Prepare data for LSTM using the same function
        x_train_stock, y_train_stock, scaler_stock, training_data_len_stock = prepare_lstm_data(stock_data, 'Close')

        # Make predictions using the loaded model
        predictions_stock = predict_with_lstm(model, stock_data, 'Close', training_data_len_stock)

        # Prepare for plotting
        train_stock = stock_data[:training_data_len_stock]
        valid_stock = stock_data[training_data_len_stock:].copy()
        valid_stock.loc[:, 'Predictions'] = predictions_stock

        # Plot the predictions for the entered stock and get base64 string
        base64_plot = plot_predictions_base64(train_stock, valid_stock)

#uAgents work start here

class Details(Model):
    stock:str
    start_date:datetime


stock = Agent(name = "stock", seed = "stock_name")
stock_graph =  Agent(name = "stock_graph", seed = "stock_graph")

@stock.on_event("startup")
async def get_file(ctx: Context):
    ctx.logger.info(f'Please enter any stock symbol from https://finance.yahoo.com/most-active/')
    stk = input()
    ctx.logger.info(f'Enter the start date (YYYY-MM-DD): ')
    date = input()
    try:
        dt = datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
        exit()
    result = await ctx.send(stock_graph.address,Details(stock = stk, start_date = dt))
    
@stock_graph.on_message(model=Details)
async def get_result(ctx:Context, sender:str, msg:Details):
    ctx.logger.info(msg.stock)
    ctx.logger.info(msg.start_date)
    predict_main(msg.stock, msg.start_date)
    
bureau = Bureau()
bureau.add(stock)
bureau.add(stock_graph)
if __name__ == "__main__":
    bureau.run()


