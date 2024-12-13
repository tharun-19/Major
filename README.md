# Stock Price Prediction Using Machine Learning

## Table of Contents
- [Abstract](#abstract)
- [Technologies Used](#technologies-used)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Technical Analysis](#technical-analysis)
  - [Fundamental Analysis](#fundamental-analysis)
  - [Ensemble Learning Techniques](#ensemble-learning-techniques)
  - [Sentiment Analysis](#sentiment-analysis)
- [Results](#results)
- [Conclusion](#conclusion)

## Abstract
Stock price prediction through machine learning is an emerging area of research that assists individuals and institutions in stabilizing their investment incomes. This project compares two primary stock market analysis methods: technical analysis and fundamental analysis. For technical analysis, ensemble classification using Decision Tree, K-Nearest Neighbors (KNN), and Random Forest algorithms combined with AdaBoost, Bagging, and Voting classifiers were implemented to enhance prediction accuracy. Fundamental analysis leverages the FB Prophet model integrated with sentiment analysis derived from New York Times news articles related to Apple Inc. The Voting-based ensemble classifier achieved the highest accuracy of 79.98%, demonstrating the effectiveness of technical analysis for short-term market predictions.

## Technologies Used
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - Pandas
  - NumPy
  - Scikit-learn
  - FB Prophet
  - NLTK / TextBlob (for NLP and sentiment analysis)
  - Matplotlib / Seaborn (for data visualization)
- **Machine Learning Algorithms:**
  - Decision Trees
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - AdaBoost
  - Bagging
  - Voting Classifier
- **Data Sources:**
  - [Yahoo Finance](https://finance.yahoo.com/) (Historical stock prices)
  - [New York Times API](https://developer.nytimes.com/) (News articles)

## Project Overview
The project focuses on predicting the stock price of Apple Inc. (AAPL) by analyzing historical price data and incorporating sentiment analysis from news articles. Two main approaches are employed:
1. **Technical Analysis:** Utilizes ensemble machine learning models to predict stock prices based on historical trading data.
2. **Fundamental Analysis:** Employs the FB Prophet model combined with sentiment analysis from news data to forecast stock prices.

## Dataset
### Historical Stock Prices
- **Source:** Yahoo Finance
- **Duration:** January 2017 to December 2020
- **Attributes:**
  - Date
  - Open
  - High
  - Low
  - Close
  - Adj Close
  - Volume

### News Articles
- **Source:** New York Times API
- **Keyword:** "Apple Inc" (AAPL)
- **Duration:** January 2017 to December 2020
- **Data Format:** JSON

## Methodology

### Technical Analysis
- **Algorithms Used:** Decision Tree, KNN, Random Forest
- **Ensemble Techniques:** AdaBoost, Bagging, Voting Classifier
- **Process:**
  1. **Data Pre-processing:** Cleaning and smoothing historical stock data.
  2. **Feature Selection:** Selecting relevant features (Open, High, Low, Close, Volume).
  3. **Model Training:** Training individual models and ensemble classifiers.
  4. **Prediction:** Forecasting closing prices based on trained models.

### Fundamental Analysis
- **Model Used:** FB Prophet
- **Process:**
  1. **Data Collection:** Gathering stock prices and corresponding news articles.
  2. **Sentiment Analysis:** Using NLP techniques to categorize sentiments as positive, negative, or neutral.
  3. **Feature Integration:** Combining sentiment scores with historical stock data.
  4. **Prediction:** Forecasting stock prices using the FB Prophet model.

### Ensemble Learning Techniques
- **AdaBoost:** Sequentially builds models by focusing on previously misclassified instances.
- **Bagging:** Reduces overfitting by training multiple models on random subsets of the data.
- **Voting Classifier:** Combines predictions from multiple models to improve accuracy.

### Sentiment Analysis
- **Technique:** Natural Language Processing (NLP)
- **Tool:** SentimentIntensityAnalyzer from NLTK/TextBlob
- **Process:**
  1. **Data Cleaning:** Removing noise from news articles.
  2. **Sentiment Scoring:** Assigning sentiment scores to each article.
  3. **Feature Addition:** Adding sentiment scores as features to the historical stock data.

## Results
- **FB Prophet Model:**
  - **Accuracy:** 70.76%
  - **MAE:** 0.97
  - **MSE:** 1.2
  - **R² Score:** 0.78
- **Ensemble Models:**
  - **Voting Classifier:** 79.98% Accuracy
  - **AdaBoost:** 77.63% Accuracy
  - **Bagging:** 77.13% Accuracy

The Voting Classifier outperformed other ensemble techniques and the FB Prophet model, indicating the effectiveness of combining multiple machine learning algorithms for stock price prediction.

## Conclusion
Predicting stock prices is inherently challenging due to market volatility influenced by various factors like financial events, corporate performance, and global trends. This project demonstrates that ensemble machine learning models, particularly the Voting Classifier, provide higher accuracy in short-term stock price predictions compared to traditional models and FB Prophet. Integrating sentiment analysis from news articles further enhances the prediction capabilities by incorporating fundamental factors.

---
