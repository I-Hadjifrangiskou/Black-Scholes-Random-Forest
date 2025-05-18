import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Black-Scholes formula for a stock price (S), strike price (K), time to expiration (T) in years, risk-free interest rate (r), volatility (sigma) and option type (call/put)
def black_scholes(S, K, T, r, sigma, option):

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) # Standardised distance between current stock price (S) and strike price (K) with drift term (r + 0.5σ^2) 
    d2 = d1 - sigma * np.sqrt(T) # Prob. of stock price exceeding strike at expiration

    # If option is a call
    if option == "call":

        # Call price
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # If option is a put
    else:

        # Put price
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Function to plot predictions for visualisation
def plot_predictions(actual, predicted, title):
    plt.scatter(actual, predicted, alpha = 0.5, label = title)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)],
             color = 'red', linestyle = 'dashed', label = 'Perfect Prediction')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.legend()
    plt.show()

# Main function
def main():

    # Generate option pricing data 
    np.random.seed(1)
    n_samples = 10000


    # Sampling parameters
    S     =     np.random.uniform(100, 200, n_samples)  # Stock price
    K     =     np.random.uniform(100, 200, n_samples)  # Strike price
    T     =     np.random.uniform(0.1, 2, n_samples)    # Time to maturity (years)
    sigma =     np.random.uniform(0.1, 0.5, n_samples)  # Volatility (fixed in this case)
    r     =     0.05       

    # Option prices according to Black-Scholes model
    call_prices = black_scholes(S, K, T, r, sigma, "call")
    put_prices = black_scholes(S, K, T, r, sigma, "put")


    # Create dataset to use for training and testing
    data = pd.DataFrame({
        "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
        "call_price": call_prices, "put_price": put_prices
    })

    # Split into train & test sets
    X = data[["S", "K", "T", "r", "sigma"]]
    y = data["call_price"] # Predicting call prices
    z = data["put_price"] # Predicting put prices


    X_train_call, X_test_call, y_train, y_test = train_test_split(X, y, test_size = 0.2) # test_size gives the proportion of the data used for testing, rest is used for training
    X_train_put, X_test_put, z_train, z_test = train_test_split(X, z, test_size = 0.2)

    # Train Random Forest model and predict
    rf_model_call = RandomForestRegressor(n_estimators = 100) # n_estimators gives the number of decision trees in the RF model
    rf_model_call.fit(X_train_call, y_train)
    call_predictions = rf_model_call.predict(X_test_call)

    rf_model_put = RandomForestRegressor(n_estimators = 100)
    rf_model_put.fit(X_train_put, z_train)
    put_predictions = rf_model_put.predict(X_test_put)

    # Normalized mean squared error for reference 
    nmse_call = mean_squared_error(y_test, call_predictions) / np.var(y_test)
    nmse_put = mean_squared_error(z_test, put_predictions) / np.var(z_test)
    print(f"Normalized mean squared error for call options: {nmse_call:.5f}")
    print(f"Normalized mean squared error for put options: {nmse_put:.5f}")

    # Plot predictions vs actual prices
    plot_predictions(y_test, call_predictions, "Call Option Prediction")
    plot_predictions(z_test, put_predictions, "Put Option Prediction")


if __name__ == "__main__":
    main()
