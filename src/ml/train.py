import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 

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
    features = []
    labels = []
    
    feature_cols = [
        'teddy_x', 'teddy_y', 'teddy_w', 'teddy_h', 'teddy_dx', 'teddy_dy',
        'closest_toy_x', 'closest_toy_y', 'closest_toy_dx', 'closest_toy_dy',
        'water_x', 'water_y'
    ]
    
    for i in range(len(df) - sequence_length):
        sequence = df[feature_cols].iloc[i:i + sequence_length].values
        next_state = df['teddy_state'].iloc[i + sequence_length]
        current_state = df['teddy_state'].iloc[i + sequence_length - 1]
        
        if next_state != current_state:  # Only include state changes
            features.append(sequence)
            labels.append(next_state)
    
    return np.array(features), np.array(labels)

def create_model(sequence_length, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Changed to 5 to match number of states
    ])
    return model

def train_model(data_source):
    df = pd.read_csv(data_source)
    
    state_map = {
        'none': 0,
        'teddy': 1,
        'teddy_lying': 2,
        'teddy_howling': 3,
        'teddy_play': 4
    }
    
    df['teddy_state'] = df['teddy_state'].map(state_map)
    
    scaler = StandardScaler()
    numerical_cols = ['teddy_x', 'teddy_y', 'teddy_w', 'teddy_h', 'teddy_dx', 'teddy_dy',
                     'closest_toy_x', 'closest_toy_y', 'closest_toy_dx', 'closest_toy_dy',
                     'water_x', 'water_y']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    sequence_length = 10
    X, y = prepare_sequences(df, sequence_length)
    
    y = tf.keras.utils.to_categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model(sequence_length, X.shape[2])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    _history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    model.save('teddy_state_predictor.h5')
    
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler

def predict_next_states(model, scaler, current_sequence):
    normalized_sequence = scaler.transform(current_sequence)
    sequence = normalized_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
    predictions = model.predict(sequence)[0]
    
    state_map = {
        0: "teddy",
        1: "teddy_lying",
        2: "teddy_howling",
        3: "teddy_play"
    }
    
    return {state_map[i]: float(prob) * 100 for i, prob in enumerate(predictions)}

if __name__ == '__main__':
    print("Training ML Model")

    if len(sys.argv) < 2:
        print("Usage: python train.py <filepath>")
        sys.exit(1)

    training_data_path = sys.argv[1]
    train_model(training_data_path)