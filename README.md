# Forecasting the price of cryptocurrencies 

---

### Background, motivation and 'results'
The aim of this project was to learn about Recurrent Neural Networks and investigate the benefit of hybrid models
in time series forecasting.

Cryptocurrencies and memestocks swept the world with excitement, giving in my opinion 
an outlet to get into something after another set of lockdowns were imposed. Hence no better subject for a personal 
project, especially after a personal ROI of 2567% on Doge. 

I set out to beat a commonsense baseline where the price at time `t+1` is equal to the 
price at time `t`. This proved (un)surprisingly difficult, (just about) succeeding in the end.

### Hybrid Model
I opted for a hybrid model, consisting of a feature and a target transforming models.
For the former I chose to use a Long Short-Term Memory network from the Keras library due to its 
inherent function for sequenced data. It also doesn't suffer from the vanishing gradient problem 
like the `SimpleRNN` layer also available from Keras. 

For the target-transforming algorithm which trained on the residuals from the LSTM, 
I chose the XGBoost from Sklearn which is a gradient boosted ensemble of decision
trees, mainly due to its speed.  



### Structure of the model

In [btc_eth_EDA.ipynb](https://github.com/mxury/crypto/blob/main/btc_eth_EDA.ipynb) we have the
exploratory data analysis. Missing values are imputed, and general statistical observables are calculated.

In [cross_validiation_split.py](https://github.com/mxury/crypto/blob/main/cross_validiation_split.py) the 
validation/training split Class is defined. Due to the time causality of the data, it is obviously
not randomised. Unlike with the standard time series split that Sklearn provides, each split here does not share 
any data with any other split. 

In [feature_engineering.ipynb](https://github.com/mxury/crypto/blob/main/feature_engineering.ipynb) the features for both models are calculated. 
The LSTM is only fed features that pertain to the longer-term nature of the time series, while the XGBoost is only trained
on features that concern the short-term volatility.

The architecture of the LSTM network can be found in [lstm.py](https://github.com/mxury/crypto/blob/main/lstm.py)
while the data preparation for it can be found in [lstm_data_gen.py](https://github.com/mxury/crypto/blob/main/lstm_data_gen.py), which 
includes functionality for debugging which can be quite cumbersome due to the batching that is needed for the LSTM. 

Lastly the XGBoost model and its architecture can be found in [xgboost_starter.ipynb](https://github.com/mxury/crypto/blob/main/xgboost_starter.ipynb).

### Lessons learned 
Optimal LSTM architecture needs to be properly investigated. Difficulty comes from model run time which severely impairs 
comparisons.

Some sentiment analysis would be handy I think as financial time series at the end of the day comes down to information arbitrage,
and in the crypto world there are no "fundamentals" and the sway of the public mood is seemingly a huge factor (not that it's not in conventional asset classes).
