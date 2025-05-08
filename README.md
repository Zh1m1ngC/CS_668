# CS_668

In this project, I am trying to find if there are any correlations between traditional stock indices like the S&P 500, U.S. Total Market Index, and Bitcoin, because they are all popular investment products (stock indices are like a basket of a group of companies, providing a good estimation of the general market trend). If there are such correlations, I want to identify which models can best predict this movement.

For this project, I used time-series models (ARIMA, Holt-Winters) and machine learning (Random Forest) and analyzed minute-level data from Polygon.io (an open-source site providing free market data).

Because I wanted to check if there are any links between the opening hours (9:30–11:30 AM) of the stock market and the overnight (8:00 PM–9:00 PM) performance of Bitcoin, then use the models mentioned to find the trends. After that, parameter tuning was applied to optimize model performance, and an LLM model was used to summarize my results.

Overall, both VTI and VOO analyses showed a training R² around 0.72, with testing R² being negative, which could mean data overfitting, data imbalance when splitting training and testing data, or some other data processing errors. However, we are currently under some very unusual market conditions where tariffs were being changed daily for a short period of time, resulting in great market volatility. Expanding the dataset to include global indices and a longer time frame could probably further validate this theory.
