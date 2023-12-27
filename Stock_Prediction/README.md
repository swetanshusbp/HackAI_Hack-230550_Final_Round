# Stock Prediction
### Project Description
The Stock Prediction project is a Python-based application that analyzes stock trends and predicts future movements. It utilizes machine learning models trained on historical stock data. The application requires users to input valid stock tickers, which are then used to fetch data and perform predictions.

### Prerequisites

Python (3.6 or higher recommended)
Required Python Libraries: numpy, pandas, scikit-learn
Access to stock data, preferably from Yahoo Finance Trending Tickers
Installation Instructions
Clone the Repository:
Clone the project repository to your local machine.

### Install Python Libraries and Dependencies:

1. After opening the terminal with its current directory at root, navigate to Stock_Prediction > uAgents > python and type and enter the following commands sequentially:

    ## poetry install
    ## poetry shell

2. After the above step, please navigate to Stock_Prediction > uAgents > python > src > agents and type and enter the following commands sequentially:

    ## pip install -r requirements.txt
    ## pip install -r requirements1.txt

    ### Please note that two requirements.txt files are required due to conflicting versions of the tensorflow and keras modules due to the version differences between the machines on which it was trained and which it was integrated into the repository

3. After the above step, assuming you are still in the current working directory of agents, hit the following command in terminal:

    ## python client.py

4. Enjoy and have a good day


pip install numpy pandas scikit-learn
Running the Project
### Fetch Stock Data:

Fetch stock data from a reliable source, such as Yahoo Finance(https://finance.yahoo.com/trending-tickers/?.tsrc=fin-srch).
Make sure to use valid stock tickers as input.

#### Using the Application:

Input the stock ticker symbols as instructed by the script.
The script will fetch the data, perform analysis, and output predictions.

### Special Considerations

Ensure that you have a stable internet connection to fetch stock data.
The accuracy of predictions depends on the quality and recency of the stock data used.

### Use cases:

    