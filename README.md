# Team Name: The Three Coders
## Team ID: Hack-230550
### Team Details:
TEAM ID  - Tech-230306
TEAM NAME: Tensor Tamers
COLLEGE: SRM INSTITUTE OF SCIENCE AND TECHNOLOGY, 
LOCATION: KATTANKULATHUR

Swetanshu Agrawal - Team Lead <br>
Email - sa0029@srmist.edu.in <br>
Phone no. - +91 7894035770 <br>
Year-3rd Year <br>
Specialization or Major: CSE with Specialization in AIML <br>
LinkedIn Id:https://www.linkedin.com/in/swetanshu-agrawal/ <br>

Samudra Banerjee <br>
Email - ss9148@srmist.edu.in <br>
Phone no. - +91 9903382548 <br>
Year-3rd Year <br>
Specialization or Major: CSE with Specialization in Software <br>
LinkedIn:https://www.linkedin.com/in/samudra-banerjee-9b4466217/ <br>


Ritveek Rana <br>
Email - rr8499@srmist.edu.in <br>
Phone no. - +91 9382756045 <br>
Year-3rd Year <br>
Specialization or Major: CSE with Specialization in AIML <br>
LinkedIn Id: https://www.linkedin.com/in/ritveekrana/ <br>

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


# Resume Screening and Classification Using Machine Learning

## Overview
This project focuses on the automated screening and classification of resumes using machine learning techniques. It utilizes Natural Language Processing (NLP) to analyze resume content and classify them into various categories based on skills, experience, and other relevant criteria. The primary goal is to assist HR departments and recruiters in efficiently sorting through large volumes of resumes.

## Model
The core of the project is a machine learning model built using Python's scikit-learn library. The model employs a text classification approach, where resumes are first preprocessed to extract relevant features, then classified using algorithms like K-Nearest Neighbors (KNN) in a One-vs-Rest (OvR) framework.

## Thought Process
Data preprocessing involves cleaning the resume text, removing unnecessary characters, and converting the text into a format suitable for machine learning (using TF-IDF vectorization).
The KNN algorithm is chosen for its simplicity and effectiveness in handling multi-class classification problems.
The model is trained on a labeled dataset of resumes, where each resume is tagged with its corresponding category (like Data Science, Web Development, etc.).
The system includes functionality to process PDF resumes, extracting text for analysis.

## Improvements
Explore advanced NLP techniques and models, such as word embeddings or deep learning-based text classifiers.
Implement additional features like work experience duration, educational background parsing, and skill extraction.
Enhance the preprocessing step to handle more complex resume formats and layouts.

## Disclaimer
This project is intended as a tool to assist in the initial stages of the resume screening process. It is not designed to replace human judgment and decision-making in recruitment. The effectiveness of the model depends on the quality and diversity of the training dataset. Users should be aware of potential biases in the dataset and use the tool accordingly. The creators of this project are not responsible for any decisions made based on the model's output.

## A wonderful journey of fetch.AI:

Gotta say, the entire 12 hours of fetch.AI competition was a good exercise to get us moving again after having not touched any development related for more than a month. While we are at this, let us document our journey throughout this 12 hours of developer gymnastics :

1. Never touched anything like uAgents library before, so it was imperative for us to thouroughly read the docs and understand the syntax and use cases as fast as possible

2. Local testing and modification of modules before integration with other parts of the project

3. Two sweet CCD sundaes along with lots of sugar rich CCD delicacies. Experienced sugar high and crash like never before, so got stalled for two hours.

4. Thought of creating multiple different util files to deal with the agents, however seeing the simplicity of the solution, and our personal preferences to stick to as many least terminals as possible, integrated the final prediction model and agents into one file

5. At first, we thought of integrating with Class modules in python. Preferred much more simplicity so stuck to the above step.

6. Woo, integrating the models into the uAgents eco-system was a nice and refreshing challenge ... we would love to attend more of this!


