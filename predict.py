import numpy as np

def make_prediction(model, X, scaler):
    last_input = X[-1].reshape(1, -1)
    predicted_scaled = model.predict(last_input, verbose=0)[0][0]

    # Only inverse transform 'Close' column (assumed to be at index 0)
    dummy = np.zeros((1, X.shape[1]))
    dummy[0][0] = predicted_scaled
    predicted_price = scaler.inverse_transform(dummy)[0][0]
    return predicted_price
