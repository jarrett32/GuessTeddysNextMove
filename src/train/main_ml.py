import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

print(tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

cols = [
    "frame_idx", "teddy_state", "teddy_confidence", "teddy_x", "teddy_y",
    "teddy_w", "teddy_h", "teddy_dx", "teddy_dy", "closest_toy_confidence",
    "closest_toy_x", "closest_toy_y", "closest_toy_w", "closest_toy_h",
    "closest_toy_dx", "closest_toy_dy", "water_confidence", "water_x",
    "water_y", "water_w", "water_h",
]

def normalize_dx_dy(number, fps):
    return number / fps

def prepare_sequences(df, sequence_length=10):
    """Prepare sequences of observations for LSTM"""
    features = []
    labels = []
    
    # Select relevant features
    feature_cols = [
        'teddy_x', 'teddy_y', 'teddy_w', 'teddy_h', 'teddy_dx', 'teddy_dy',
        'closest_toy_x', 'closest_toy_y', 'closest_toy_dx', 'closest_toy_dy',
        'water_x', 'water_y'
    ]
    
    for i in range(len(df) - sequence_length):
        # Get sequence of observations
        sequence = df[feature_cols].iloc[i:i + sequence_length].values
        # Get next state (target)
        next_state = df['teddy_state'].iloc[i + sequence_length]
        # Get current state to exclude it from target possibilities
        current_state = df['teddy_state'].iloc[i + sequence_length - 1]
        
        if next_state != current_state:  # Only include state changes
            features.append(sequence)
            labels.append(next_state)
    
    return np.array(features), np.array(labels)

def create_model(sequence_length, n_features):
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')  # Changed to 5 to match number of states
    ])
    return model

def train_model(data_source):
    # Read the CSV file
    df = pd.read_csv(data_source)
    
    # Update state mapping to start from 0
    state_map = {
        'none': 0,
        'teddy': 1,
        'teddy_lying': 2,
        'teddy_howling': 3,
        'teddy_play': 4
    }
    
    # Convert states to numbers using the mapping
    df['teddy_state'] = df['teddy_state'].map(state_map)
    
    # Normalize the numerical columns
    scaler = StandardScaler()
    numerical_cols = ['teddy_x', 'teddy_y', 'teddy_w', 'teddy_h', 'teddy_dx', 'teddy_dy',
                     'closest_toy_x', 'closest_toy_y', 'closest_toy_dx', 'closest_toy_dy',
                     'water_x', 'water_y']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Prepare sequences
    sequence_length = 10  # Use last 10 frames to predict next state
    X, y = prepare_sequences(df, sequence_length)
    
    # Remove the subtraction since states are already 0-3
    y = tf.keras.utils.to_categorical(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    model = create_model(sequence_length, X.shape[2])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Save the model
    model.save('teddy_state_predictor.h5')
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler

def predict_next_states(model, scaler, current_sequence):
    """
    Predict probabilities of next states
    Returns: probabilities for each state transition
    """
    # Normalize the sequence
    normalized_sequence = scaler.transform(current_sequence)
    # Reshape for model input
    sequence = normalized_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
    # Get predictions
    predictions = model.predict(sequence)[0]
    
    # Update state map to match new mapping
    state_map = {
        0: "teddy",
        1: "teddy_lying",
        2: "teddy_howling",
        3: "teddy_play"
    }
    
    return {state_map[i]: float(prob) * 100 for i, prob in enumerate(predictions)}

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python main_ml.py <filepath>")
        sys.exit(1)

    training_data_path = sys.argv[1]
    train_model(training_data_path)

# Training
# smodel, scaler = train_model('your_data.csv')

# Making predictions (example)
# sequence = get_last_10_frames()  # You'll need to implement this
# predictions = predict_next_states(model, scaler, sequence)
# print(predictions)
# Output example:
# {
#     'teddy': 10.5,
#     'teddy_lying': 60.2,
#     'teddy_howling': 20.1,
#     'teddy_play': 9.2
# }
