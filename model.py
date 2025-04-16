from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

def prepare_data(df):
    data = df[['High']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_and_predict_model(X, y, scaler):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict the next day
    last_60 = X[-1].reshape(1, X.shape[1], 1)
    pred_scaled = model.predict(last_60, verbose=0)
    prediction = scaler.inverse_transform(pred_scaled)[0][0]

    return model, prediction
