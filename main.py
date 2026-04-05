import numpy as np, pandas as pd, matplotlib.pyplot as plt, os

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score


def relative_error(y_true, y_pred):
    # the ys could be inverted, it would not make a difference becaues we take the abs
    return np.mean(np.abs((y_true - y_pred) / y_true))

def print_eval(X, y, model):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    re = relative_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"Mean squared error:\t\t {mse:.5}")
    print(f"Relative error: \t\t {re:.5%}")
    print(f"R-squared coefficient:\t {r2:.5}")
    
def plot_model_on_data(X, y, model=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(X, y)
    if model is not None:
        xlim, ylim = plt.xlim(), plt.ylim()
        line_x = np.linspace(xlim[0], xlim[1], 100)
        line_x_df = pd.DataFrame(line_x[:, None], columns=X.columns)
        line_y = model.predict(line_x_df)
        plt.plot(line_x, line_y, c="red", lw=3)
        plt.xlim(xlim); plt.ylim(ylim)
    plt.grid()
    plt.xlabel("Temperatura (°C)"); plt.ylabel("Consumi (GW)")

def main():
    
    POWER_DATA_URL = "https://github.com/datascienceunibo/dialab2024/raw/main/Regressione_non_Lineare/power.csv"
    if not os.path.exists("power.csv"):
        from urllib.request import urlretrieve
        urlretrieve(POWER_DATA_URL, "power.csv")
        
    power = pd.read_csv("power.csv", index_col="date", parse_dates=["date"])
    
    # print(power.head(8))
    
    power_summer = power.loc[power.index.month.isin([6, 7, 8])]
    
    # print(power_summer)
    
    lrm = LinearRegression()
    
    # Gives us a DataFrame instead of a series
    X = power_summer[["temp"]]
    y = power_summer["demand"]
    
    # returns a model that we can use, in this case linear regression model
    lrm.fit(X, y)
    
    # preds = lrm.predict(X)
    
    # returns a numpy array, if we want a pandas series we need to do it manually
    # print(preds[:5])
    
    preds = pd.Series(lrm.predict(X), index=power_summer.index)
    
    # print(preds.head())
    
    # To get alpha and beta used in the formula alpha * x + beta we can access them 
    
    # print(lrm.coef_, lrm.intercept_)
    
    manual_error = np.mean(np.square(preds - y))
    
    sk_error = mean_squared_error(y, preds)
    
    
    # Sks error will always be <= gd 
    # this is because sk uses a formula, 
    # this is the best mathematical error you can find
    # gd uses a descent and iterative formula,
    # we still use gd becuase it scales better,
    # analitical way inverts a matrix which gets VERY expensive
    # print(f"Manual Error = {manual_error}\nSk Error = {sk_error}\ndifference => {manual_error - sk_error}")
    
    # print(relative_error(y, preds))
    
    # The dumbest possible model is "always predict the average"
    # R² measures how much better you're doing than that.
    # print(r2_score(y, preds))
    
    # print(print_eval(power_summer[["temp"]], power_summer["demand"], lrm))
    
    # plot_model_on_data(X, y, lrm)
    
    # it is possible to add modifiers to models during traning, called hyperparameters 
    # One of these is fit_intercept for linear reg which dictates where the intercept ( interecetta )
    # is to be calulated or not #
    
    lrm_ni = LinearRegression(fit_intercept=False)
    
    lrm_ni.fit(X, y)
    
    # print(print_eval(X, y, lrm_ni))
    
    # plt.figure(figsize=(12, 8))
    # plt.scatter(X, y)
    # plt.scatter(0, 0, s=100, c="red")
    # line_x = np.linspace(-2, 32, 100)
    # line_x_df = pd.DataFrame(line_x[:, None], columns=X.columns)
    # line_y = lrm.predict(line_x_df)
    # plt.plot(line_x, line_y, c="green", lw=2)
    # line_y0 = lrm_ni.predict(line_x_df)
    # plt.plot(line_x, line_y0, c="red", lw=2)
    # plt.legend(["Dati", "Origine", "Modello con intercetta", "Modello senza intercetta"])
    # plt.xlim((-2, 32))
    # plt.ylim((-0.2, 2.8))
    # plt.grid()
    # plt.xlabel("Temperatura (°C)")
    # plt.ylabel("Consumi (GW)")
    
    is_train = power_summer.index.year < 2016
    
    # ~is_train sets all values to false so takes all values that are not in is_train
    summer_train = power_summer.loc[is_train]
    summer_test = power_summer.loc[~is_train]
    
    summer_X_train = summer_train[["temp"]]
    summer_y_train = summer_train["demand"]
    summer_X_test = summer_test[["temp"]]
    summer_y_test = summer_test["demand"]
    
    lrm = LinearRegression()
    
    lrm.fit(summer_X_train, summer_y_train)
    
    print(print_eval(summer_X_train, summer_y_train, lrm))
    
    print(print_eval(summer_X_test, summer_y_test, lrm))
    
    # plot_model_on_data(summer_X_train, summer_y_train, lrm)
    plot_model_on_data(summer_X_test, summer_y_test, lrm)
    
    plt.show()
    
if __name__ == "__main__":
    main()
