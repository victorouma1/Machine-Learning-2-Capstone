import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import keras_tuner as kt
from datetime import timedelta
import streamlit as st

import os
import random
import tensorflow as tf

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

seed_everything(42)

data = pd.read_csv('AEP_hourly.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace = True)

st.title('Electricity Consumption Forecasting')

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["EDA and Cleaning", "SARIMA/ARIMA","Prophet", "Deep Learning","Base ML","Tuning","Forecast"])

with tab1:
    st.subheader('121273 obsservations')
    fig, ax = plt.subplots()
    sns.histplot(data['AEP_MW'], kde = True)
    st.pyplot(fig)
    st.write("Data is normally distributed so Standard Scaler would be most suitable")
    fig_2,ax_2 = plt.subplots()
    data_2 = data[['AEP_MW']].resample('ME').mean().copy()
    data_2['AEP_MW'].plot(title=f"Consumption over time", figsize=(20, 5))
    st.pyplot(fig_2)
    data_time = data.resample('D').mean().copy()
    st.header('Train Test split')
    st.write("80-20 split betwwen testing and training data")
    size = int(len(data_time)*0.8)
    train_dataset = data_time.iloc[:size].copy()
    test_dataset = data_time.iloc[size:].copy()
    st.header('Checking for stationarity')
    st.write(f"An ADF test is done and the p_value obatined is: {sts.adfuller(train_dataset['AEP_MW'])[1]}"
             " which is less than 0.05 so it rejects the null hypothesis meaning series is stationary")
    st.header('Sesonal Decompose')
    s_dec_additive = seasonal_decompose(train_dataset['AEP_MW'], model='additive')
    fig_3 = s_dec_additive.plot()
    st.pyplot(fig_3)
    fig_4= sgt.plot_acf(train_dataset['AEP_MW'], lags=50)
    plt.title("AEP (MegaWatts) ACF", size=24)    
    st.pyplot(fig_4)
    fig_5 = sgt.plot_pacf(train_dataset['AEP_MW'], lags=50, method='ols')
    plt.title("AEP (MegaWatts) PACF", size=24)
    st.pyplot(fig_5)

with tab2:
    st.header("SARIMA")
    st.write("Done with order (2,0,0) and seaonal order (1,0,1,7)")
    model_sarima = SARIMAX(train_dataset['AEP_MW'],
                       order=(2, 0, 0),           
                       seasonal_order=(1, 0, 1, 7),
                       enforce_stationarity=False,
                       enforce_invertibility=False)

    results_sarima = model_sarima.fit()
    n_steps = len(test_dataset)
    forecast_obj = results_sarima.get_forecast(steps=n_steps)

    y_pred = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    y_pred_train = results_sarima.fittedvalues

    fig_6 = plt.figure(figsize=(10,5))
    plt.plot(test_dataset.index, test_dataset['AEP_MW'], label='Test', color='orange')
    plt.plot(y_pred.index, y_pred, label='Forecast', color='green')
    plt.fill_between(conf_int.index,
                    conf_int.iloc[:,0],
                    conf_int.iloc[:,1],
                    color='green', alpha=0.2)
    plt.legend()
    plt.title('SARIMA(2,0,0) Forecast vs Actual')
    st.pyplot(fig_6)

    mae = mean_absolute_error(test_dataset['AEP_MW'], y_pred)
    rmse = np.sqrt(mean_squared_error(test_dataset['AEP_MW'], y_pred))
    mae_train = mean_absolute_error(train_dataset['AEP_MW'], y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(train_dataset['AEP_MW'], y_pred_train))
    mape_train = mean_absolute_percentage_error(train_dataset['AEP_MW'], y_pred_train)
    mape_test  = mean_absolute_percentage_error(test_dataset['AEP_MW'], y_pred)
    train_r2 = r2_score(train_dataset['AEP_MW'], y_pred_train)
    test_r2 = r2_score(test_dataset['AEP_MW'], y_pred)

    st.write(f"Train MAE : {mae_train:.2f}")
    st.write(f"Test MAE : {mae:.2f}")
    st.write("")
    st.write(f"Train RMSE: {rmse_train:.2f}")
    st.write(f"Test RMSE: {rmse:.2f}")
    st.write("")
    st.write(f"Train MAPE: {mape_train * 100:.2f}%")
    st.write(f"Test MAPE:  {mape_test * 100:.2f}%")
    st.write("")
    st.write(f"Train R2: {train_r2:.2f}")
    st.write(f"Test R2: {test_r2:.2f}")

    st.header('ARIMA')

    model_arima = ARIMA(train_dataset['AEP_MW'], order=(2,0,0))
    results_arima = model_arima.fit()

    forecast_obj_1 = results_arima.get_forecast(steps=n_steps)

    y_pred_1 = forecast_obj_1.predicted_mean
    conf_int_1 = forecast_obj_1.conf_int()

    y_pred_train_1 = results_arima.fittedvalues

    fig_7 = plt.figure(figsize=(10,5))
    plt.plot(test_dataset.index, test_dataset['AEP_MW'], label='Test', color='orange')
    plt.plot(y_pred_1.index, y_pred_1, label='Forecast', color='green')
    plt.fill_between(conf_int_1.index,
                    conf_int_1.iloc[:,0],
                    conf_int_1.iloc[:,1],
                    color='green', alpha=0.2)
    plt.legend()
    plt.title('ARIMA(2,0,0) Forecast vs Actual')
    st.pyplot(fig_7)

    mae_1 = mean_absolute_error(test_dataset['AEP_MW'], y_pred_1)
    rmse_1 = np.sqrt(mean_squared_error(test_dataset['AEP_MW'], y_pred_1))
    mae_train_1 = mean_absolute_error(train_dataset['AEP_MW'], y_pred_train_1)
    rmse_train_1 = np.sqrt(mean_squared_error(train_dataset['AEP_MW'], y_pred_train_1))
    train_r2_1 = r2_score(train_dataset['AEP_MW'], y_pred_train_1)
    test_r2_1 = r2_score(test_dataset['AEP_MW'], y_pred_1)
    mape_train_1 = mean_absolute_percentage_error(train_dataset['AEP_MW'], y_pred_train_1)
    mape_test_1  = mean_absolute_percentage_error(test_dataset['AEP_MW'], y_pred_1)

    st.write(f"Train MAE : {mae_train_1:.2f}")
    st.write(f"Test MAE : {mae_1:.2f}")
    st.write("")
    st.write(f"Train RMSE: {rmse_train_1:.2f}")
    st.write(f"Test RMSE: {rmse_1:.2f}")
    st.write("")
    st.write(f"Train MAPE: {mape_train_1*100:.2f}%")
    st.write(f"Test MAPE:  {mape_test_1*100:.2f}%")
    st.write("")
    st.write(f"Train R2: {train_r2_1:.2f}")
    st.write(f"Test R2: {test_r2_1:.2f}")

    st.header('Residuals')
    arima_resid = results_arima.resid
    sarima_resid = results_sarima.resid
    c1, c2 = st.columns(2)
    with c1:
        fitted_values = results_sarima.fittedvalues

        fig_8 = plt.figure(figsize=(8, 6))
        plt.scatter(fitted_values, results_sarima.resid, alpha=0.5, color='darkblue', s=15)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted (Predicted) Values')
        plt.ylabel('Residuals')
        plt.title('SARIMA Residuals vs Fitted Values')
        st.pyplot(fig_8)
    with c2:
        fitted_values_1 = results_arima.fittedvalues

        fig_9 = plt.figure(figsize=(8, 6))
        plt.scatter(fitted_values_1, results_arima.resid, alpha=0.5, color='darkblue', s=15)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted (Predicted) Values')
        plt.ylabel('Residuals')
        plt.title('ARIMA Residuals vs Fitted Values')
        st.pyplot(fig_9)

with tab3:
    train_proph = train_dataset.copy()
    train_proph.reset_index(inplace = True)
    train_proph = train_proph.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})
    model = Prophet()
    model.fit(train_proph)
    test_proph = test_dataset.copy()
    test_proph.reset_index(inplace = True)
    test_proph = test_proph.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})
    forecast = model.predict(test_proph)
    forecast_train = model.predict(train_proph)

    fig_10 = plt.figure(figsize=(10,5))
    plt.plot(test_proph['ds'], test_proph['y'], label='Test', color='orange')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
    plt.fill_between(forecast['ds'],
                    forecast['yhat_upper'],
                    forecast['yhat_lower'],
                    color='green', alpha=0.2)
    plt.legend()
    plt.title('Prophet Forecast vs Actual')
    st.pyplot(fig_10)

    mae_2 = mean_absolute_error(test_proph['y'], forecast['yhat'])
    rmse_2 = np.sqrt(mean_squared_error(test_proph['y'], forecast['yhat']))
    mae_train_2 = mean_absolute_error(train_proph['y'], forecast_train['yhat'])
    rmse_train_2 = np.sqrt(mean_squared_error(train_proph['y'], forecast_train['yhat']))
    train_r2_2 = r2_score(train_proph['y'], forecast_train['yhat'])
    test_r2_2 = r2_score(test_proph['y'], forecast['yhat'])
    mape_test_2 = mean_absolute_percentage_error(test_proph['y'],forecast['yhat'])
    mape_train_2 = mean_absolute_percentage_error(train_proph['y'], forecast_train['yhat'])
    st.write(f"Train MAE : {mae_train_2:.2f}")
    st.write(f"Test MAE : {mae_2:.2f}")
    st.write("")
    st.write(f"Train RMSE: {rmse_train_2:.2f}")
    st.write(f"Test RMSE: {rmse_2:.2f}")
    st.write("")
    st.write(f"Train MAPE: {mape_train_2*100:.2f}%")
    st.write(f"Test MAPE:  {mape_test_2*100:.2f}%")
    st.write("")
    st.write(f"Train R2: {train_r2_2:.2f}")
    st.write(f"Test R2: {test_r2_2:.2f}")


with tab4:
    st.header("RNN")
    data_3 = data_time.copy()
    window_size = 12

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_3)

    X = []
    y = []
    target_dates = data_3.index[window_size:]

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, target_dates, test_size=0.2, shuffle=False, random_state = 42
    )

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(SimpleRNN(units=128,activation = 'relu',return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.35))
    model.add(SimpleRNN(units=128,activation = 'relu'))
    model.add(Dropout(0.35))
    model.add(Dense(1))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    custom_learning_rate = 0.001
    optimizer = Adam(learning_rate=custom_learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = model.fit(X_train, 
                    y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.1,
                    callbacks=[early_stop])

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions).flatten()
    y_test_1 = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    predictions_train = model.predict(X_train)
    predictions_train = scaler.inverse_transform(predictions_train).flatten()
    y_train_1 = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()

    mae_3 = mean_absolute_error(y_test_1,predictions)
    rmse_3 = np.sqrt(mean_squared_error(y_test_1,predictions))
    mae_train_3 = mean_absolute_error(y_train_1,predictions_train)
    rmse_train_3 = np.sqrt(mean_squared_error(y_train_1,predictions_train))
    train_r2_3= r2_score(y_train_1,predictions_train)
    test_r2_3 = r2_score(y_test_1, predictions)
    mape_train_3 = mean_absolute_percentage_error(y_train_1,predictions_train)
    mape_test_3  = mean_absolute_percentage_error(y_test_1,predictions)

    st.write(f"Train MAE : {mae_train_3:.2f}")
    st.write(f"Test MAE : {mae_3:.2f}")
    st.write(f"Train RMSE: {rmse_train_3:.2f}")
    st.write(f"Test RMSE: {rmse_3:.2f}")
    st.write(f"Train MAPE: {mape_train_3*100:.2f}%")
    st.write(f"Test MAPE:  {mape_test_3*100:.2f}%")
    st.write(f"Train R2 Score: {train_r2_3}")
    st.write(f"Test R2 Score: {test_r2_3}")

    fig_11 = plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test_1, label='Actual Consumption')
    plt.plot(dates_test, predictions, label='Predicted Comsumption')
    plt.title('Actual vs Predicted Comsumption MegaWatts')
    plt.xlabel('Date')
    plt.ylabel('Power Comsumption (MW)')
    plt.legend()
    st.pyplot(fig_11)

    fig_12,ax = plt.subplots()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_12)

    st.header('LSTM')

    model_1 = Sequential()
    model_1.add(LSTM(units=128,activation = 'relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_1.add(Dropout(0.35))
    model_1.add(LSTM(units=128, activation = 'relu'))
    model_1.add(Dropout(0.35))
    model_1.add(Dense(1))

    early_stop_1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    optimizer_1 = Adam(learning_rate=custom_learning_rate)
    model_1.compile(optimizer=optimizer_1, loss='mean_squared_error')

    history_1 = model_1.fit(X_train, 
                    y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.1,
                    callbacks=[early_stop_1])

    predictions_1 = model_1.predict(X_test)
    predictions_1 = scaler.inverse_transform(predictions_1).flatten()

    predictions_train_1 = model_1.predict(X_train)
    predictions_train_1 = scaler.inverse_transform(predictions_train_1).flatten()

    mae_4 = mean_absolute_error(y_test_1,predictions_1)
    rmse_4 = np.sqrt(mean_squared_error(y_test_1,predictions_1))
    mae_train_4 = mean_absolute_error(y_train_1,predictions_train_1)
    rmse_train_4 = np.sqrt(mean_squared_error(y_train_1,predictions_train_1))
    train_r2_4= r2_score(y_train_1,predictions_train_1)
    test_r2_4 = r2_score(y_test_1, predictions_1)
    mape_train_4 = mean_absolute_percentage_error(y_train_1,predictions_train_1)
    mape_test_4  = mean_absolute_percentage_error(y_test_1,predictions_1)

    st.write(f"Train MAE : {mae_train_4:.2f}")
    st.write(f"Test MAE : {mae_4:.2f}")
    st.write(f"Train RMSE: {rmse_train_4:.2f}")
    st.write(f"Test RMSE: {rmse_4:.2f}")
    st.write(f"Train MAPE: {mape_train_4*100:.2f}%")
    st.write(f"Test MAPE:  {mape_test_4*100:.2f}%")
    st.write(f"Train R2 Score: {train_r2_4}")
    st.write(f"Test R2 Score: {test_r2_4}")

    fig_13 = plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test_1, label='Actual Consumption')
    plt.plot(dates_test, predictions_1, label='Predicted Comsumption')
    plt.title('Actual vs Predicted Comsumption MegaWatts')
    plt.xlabel('Date')
    plt.ylabel('Power Comsumption (MW)')
    plt.legend()
    st.pyplot(fig_13)

    fig_14,ax = plt.subplots()
    plt.plot(history_1.history['loss'], label='Training Loss')
    plt.plot(history_1.history['val_loss'], label='Validation Loss')

    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_14)


    st.header('GRU')

    model_2 = Sequential()
    model_2.add(GRU(units=128,activation = 'relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_2.add(Dropout(0.35))
    model_2.add(GRU(units=128, activation = 'relu'))
    model_2.add(Dropout(0.35))
    model_2.add(Dense(1))

    early_stop_2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    optimizer_2 = Adam(learning_rate=custom_learning_rate)
    model_2.compile(optimizer=optimizer_2, loss='mean_squared_error')

    history_2 = model_2.fit(X_train, 
                    y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.1,
                    callbacks=[early_stop_2])

    predictions_2 = model_2.predict(X_test)
    predictions_2 = scaler.inverse_transform(predictions_2).flatten()

    predictions_train_2 = model_2.predict(X_train)
    predictions_train_2 = scaler.inverse_transform(predictions_train_2).flatten()

    mae_5 = mean_absolute_error(y_test_1,predictions_2)
    rmse_5 = np.sqrt(mean_squared_error(y_test_1,predictions_2))
    mae_train_5 = mean_absolute_error(y_train_1,predictions_train_2)
    rmse_train_5 = np.sqrt(mean_squared_error(y_train_1,predictions_train_2))
    train_r2_5= r2_score(y_train_1,predictions_train_2)
    test_r2_5 = r2_score(y_test_1, predictions_2)
    mape_train_5 = mean_absolute_percentage_error(y_train_1,predictions_train_2)
    mape_test_5  = mean_absolute_percentage_error(y_test_1,predictions_2)

    st.write(f"Train MAE : {mae_train_5:.2f}")
    st.write(f"Test MAE : {mae_5:.2f}")
    st.write(f"Train RMSE: {rmse_train_5:.2f}")
    st.write(f"Test RMSE: {rmse_5:.2f}")
    st.write(f"Train MAPE: {mape_train_5*100:.2f}%")
    st.write(f"Test MAPE:  {mape_test_5*100:.2f}%")
    st.write(f"Train R2 Score: {train_r2_5}")
    st.write(f"Test R2 Score: {test_r2_5}")

    fig_15 = plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test_1, label='Actual Consumption')
    plt.plot(dates_test, predictions_2, label='Predicted Comsumption')
    plt.title('Actual vs Predicted Comsumption MegaWatts')
    plt.xlabel('Date')
    plt.ylabel('Power Comsumption (MW)')
    plt.legend()
    st.pyplot(fig_15)

    fig_16,ax = plt.subplots()
    plt.plot(history_2.history['loss'], label='Training Loss')
    plt.plot(history_2.history['val_loss'], label='Validation Loss')

    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_16)

with tab5:
    st.header("Random Forest")
    st.write("Split date into day, month and year columns and engineered lag-1,lag-2,lag-7 columns")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth = 3)

    ML_train = train_proph.copy()
    ML_test = test_proph.copy()

    ML_train['year'] = ML_train['ds'].dt.year
    ML_train['month'] = ML_train['ds'].dt.month
    ML_train['day'] = ML_train['ds'].dt.day
    ML_train['dayofweek'] = ML_train['ds'].dt.dayofweek   
    ML_train['dayofyear'] = ML_train['ds'].dt.dayofyear   
    ML_train['is_weekend'] = ML_train['dayofweek'].isin([5, 6]).astype(int)
    ML_train = ML_train.dropna().reset_index(drop=True)
    ML_train.drop(columns = ['ds'], inplace = True)

    ML_test['year'] = ML_test['ds'].dt.year
    ML_test['month'] = ML_test['ds'].dt.month
    ML_test['day'] = ML_test['ds'].dt.day
    ML_test['dayofweek'] = ML_test['ds'].dt.dayofweek   
    ML_test['dayofyear'] = ML_test['ds'].dt.dayofyear   
    ML_test['is_weekend'] = ML_test['dayofweek'].isin([5, 6]).astype(int)
    ML_test = ML_test.dropna().reset_index(drop=True)
    ML_test.drop(columns = ['ds'], inplace = True)

    
    combined = pd.concat([ML_train, ML_test]).reset_index(drop=True)
    combined['lag_1'] = combined['y'].shift(1)
    combined['lag_2'] = combined['y'].shift(2)
    combined['lag_7'] = combined['y'].shift(7)
    combined = combined.dropna()

    size_1 = int(len(combined)*0.8)
    ML_train = combined.iloc[:size_1].copy()
    ML_test  = combined.iloc[size_1:].copy()

    X_train_2, y_train_2 = ML_train.drop(columns = 'y'), ML_train[['y']]
    X_test_2, y_test_2 = ML_test.drop(columns = 'y'), ML_test[['y']]    

    rf_model.fit(X_train_2, y_train_2)
    train_preds_rf = rf_model.predict(X_train_2)
    test_preds_rf = rf_model.predict(X_test_2)

    st.write(f"Training MAE score: {mean_absolute_error(y_train_2, train_preds_rf)}")
    st.write(f"Testing MAE score: {mean_absolute_error(y_test_2, test_preds_rf)}")
    st.write(f"Training RMSE score: {np.sqrt(mean_squared_error(y_train_2, train_preds_rf))}")
    st.write(f"Testing RMSE score: {np.sqrt(mean_squared_error(y_test_2, test_preds_rf))}")
    st.write(f"Training R2 score: {r2_score(y_train_2, train_preds_rf)}")
    st.write(f'Testing R2 score: {r2_score(y_test_2,test_preds_rf)}')

    fig_60 = plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), y_test_2, label='Actual Consumption')
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), test_preds_rf, label='Predicted Comsumption')
    plt.title('Actual vs Predicted Comsumption MegaWatts')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (MW)')
    plt.legend()
    st.pyplot(fig_60)

    st.header('Gradient Boosting')

    gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42
    )

    gb_model.fit(X_train_2, y_train_2)

    train_preds_gb = gb_model.predict(X_train_2)
    test_preds_gb = gb_model.predict(X_test_2)

    st.write(f"Training MAE score: {mean_absolute_error(y_train_2, train_preds_gb)}")
    st.write(f"Testing MAE score: {mean_absolute_error(y_test_2, test_preds_gb)}")
    st.write(f"Training RMSE score: {np.sqrt(mean_squared_error(y_train_2, train_preds_gb))}")
    st.write(f"Testing RMSE score: {np.sqrt(mean_squared_error(y_test_2, test_preds_gb))}")
    st.write(f"Training R2 score: {r2_score(y_train_2, train_preds_gb)}")
    st.write(f'Testing R2 score: {r2_score(y_test_2,test_preds_gb)}')

    fig_50 = plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), y_test_2, label='Actual Consumption')
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), test_preds_gb, label='Predicted Comsumption')
    plt.title('Actual vs Predicted Comsumption MegaWatts')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (MW)')
    plt.legend()
    st.pyplot(fig_50)

    st.header('XGBoost')

    xgb_model = XGBRegressor(
        n_estimators = 100,
        random_state = 42,
        max_depth = 3)
    xgb_model.fit(X_train_2, y_train_2)

    test_pred_xgb = xgb_model.predict(X_test_2)
    train_pred_xgb = xgb_model.predict(X_train_2)

    st.write(f"Training MAE score: {mean_absolute_error(y_train_2, train_pred_xgb)}")
    st.write(f"Testing MAE score: {mean_absolute_error(y_test_2, test_pred_xgb)}")
    st.write(f"Training RMSE score: {np.sqrt(mean_squared_error(y_train_2, train_pred_xgb))}")
    st.write(f"Testing RMSE score: {np.sqrt(mean_squared_error(y_test_2, test_pred_xgb))}")
    st.write(f"Training R2 score: {r2_score(y_train_2, train_pred_xgb)}")
    st.write(f'Testing R2 score: {r2_score(y_test_2,test_pred_xgb)}')

    fig_40 = plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), y_test_2, label='Actual Consumption')
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), test_pred_xgb, label='Predicted Comsumption')
    plt.title('Actual vs Predicted Comsumption MegaWatts')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (MW)')
    plt.legend()
    st.pyplot(fig_40)

with tab6:
    st.header("XGboost Tuned")
    xgb_model_1 = XGBRegressor(tree_method="hist", random_state=42)
    param_distributions = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01,0.02,0.03,0.04,0.05],
    "n_estimators": [300,310,320,330,340,350],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [9.0,10.0,11.0,12.0,13.0], 
    "reg_alpha": [5.0,6.0,7.0,8.0,9.0],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
    estimator=xgb_model_1,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    )
    random_search.fit(X_train_2, y_train_2)
    best_xgb_model = random_search.best_estimator_

    y_train_pred = best_xgb_model.predict(X_train_2)
    y_test_pred = best_xgb_model.predict(X_test_2)

    st.write(f"Training MAE score: {mean_absolute_error(y_train_2, y_train_pred):.2f}")
    st.write(f"Testing MAE score:  {mean_absolute_error(y_test_2, y_test_pred):.2f}")
    st.write(f"Training RMSE score: {np.sqrt(mean_squared_error(y_train_2, y_train_pred)):.2f}")
    st.write(f"Testing RMSE score:  {np.sqrt(mean_squared_error(y_test_2, y_test_pred)):.2f}")
    st.write(f"Training R2 score:   {r2_score(y_train_2, y_train_pred):.2f}")
    st.write(f"Testing R2 score:    {r2_score(y_test_2, y_test_pred):.2f}")

    feat_importances_xgb = pd.Series(
    best_xgb_model.feature_importances_,
    index=X_train_2.columns
    ).sort_values()

    fig_17, ax_17 = plt.subplots()
    feat_importances_xgb.plot(kind="barh")
    plt.title("XGBoost Feature Importances")
    st.pyplot(fig_17)

    fig_30 = plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), y_test_2, label='Actual Consumption')
    plt.plot(pd.to_datetime(ML_test[['year','month','day']]), y_test_pred, label='Predicted Comsumption')
    plt.title('Actual vs Predicted Comsumption MegaWatts')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (MW)')
    plt.legend()
    st.pyplot(fig_30)

with tab7:
    full_ml = pd.concat([train_proph, test_proph], ignore_index=True)
    full_ml = full_ml.sort_values('ds').reset_index(drop=True)

    last_test_date = test_proph['ds'].max()

    history = list(zip(full_ml['ds'], full_ml['y'])) 

    future_dates  = pd.date_range(
        start=last_test_date + pd.Timedelta(days=1),
        periods=365,
        freq='D'
    )

    future_preds = []

    for future_date in future_dates:

        recent = {d: v for d, v in history[-10:]}

        lag1_date = future_date - pd.Timedelta(days=1)
        lag2_date = future_date - pd.Timedelta(days=2)
        lag7_date = future_date - pd.Timedelta(days=7)

        lag_1 = recent.get(lag1_date, np.nan)
        lag_2 = recent.get(lag2_date, np.nan)
        lag_7 = recent.get(lag7_date, np.nan)

        row = pd.DataFrame([{
            'year'       : future_date.year,
            'month'      : future_date.month,
            'day'        : future_date.day,
            'dayofweek'  : future_date.dayofweek,
            'dayofyear'  : future_date.day_of_year,
            'is_weekend' : int(future_date.dayofweek in [5, 6]),
            'lag_1'      : lag_1,
            'lag_2'      : lag_2,
            'lag_7'      : lag_7,
        }])


        row = row[X_train_2.columns]

        pred = best_xgb_model.predict(row)[0]
        future_preds.append(pred)
        history.append((future_date, pred))

    future_df = pd.DataFrame({'ds': future_dates, 'forecast': future_preds})
    print(future_df.head())
    print(f"\nForecast shape: {future_df.shape}")

    fig_18, ax_18 = plt.subplots(figsize=(18, 6))

    context = test_proph.tail(180)
    ax_18.plot(context['ds'], context['y'],
            color='steelblue', linewidth=1.2, label='Historical (test set)')

    ax_18.plot(future_df['ds'], future_df['forecast'],
            color='seagreen', linewidth=2, label='Future forecast (1 year)')


    ax_18.axvline(last_test_date, color='red', linestyle='--', linewidth=1.2,
            label='Forecast start')

    ax_18.set_title('Tuned XGBoost — 1-Year Future Forecast',
                fontsize=14)
    ax_18.set_xlabel('Date')
    ax_18.set_ylabel('Power Consumption (MW)')
    ax_18.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_18)

    




