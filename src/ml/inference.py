import time
import pandas as pd
import tensorflow as tf
import joblib

import sys

model_path = 'src/ml/models/teddy/teddy_state_predictor.h5'
scaler_path = 'src/ml/models/teddy/scaler.pkl'

priority_state = [
    {
        'STATE': 'TEDDY',
        'LABEL': 'teddy',
        'PRIORITY': 1
    },
    {
        'STATE': 'TEDDY_LYING',
        'label': 'teddy_lying',
        'PRIORITY': 2
    },
    {
        'STATE': 'TEDDY_PLAYING',
        'LABEL': 'teddy_play',
        'PRIORITY': 3
    },
    {
        'STATE': 'TEDDY_HOWLING',
        'LABEL': 'teddy_howling',
        'PRIORITY': 4
    },
]

def generate_predictions(annotation_df_path, model_path, scaler_path):
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={
        'LSTM': lambda **kwargs: tf.keras.layers.LSTM(
            kwargs.pop('units'),
            **{k: v for k, v in kwargs.items() if k != 'time_major'}
        )
    })
    scaler = joblib.load(scaler_path)
    
    df = pd.read_csv(annotation_df_path)
    
    feature_cols = [
        'teddy_x', 'teddy_y', 'teddy_w', 'teddy_h', 'teddy_dx', 'teddy_dy',
        'closest_toy_x', 'closest_toy_y', 'closest_toy_dx', 'closest_toy_dy',
        'water_x', 'water_y'
    ]
    
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    sequence_length = 10
    predictions = []
    inference_times = []
    
    for i in range(len(df) - sequence_length + 1):
        sequence = df[feature_cols].iloc[i:i + sequence_length].values
        sequence = sequence.reshape(1, sequence_length, len(feature_cols))

        start_time = time.perf_counter()
        pred_probs = model.predict(sequence, verbose=0)[0]
        inference_time = (time.perf_counter() - start_time) * 1000
        inference_times.append(inference_time)
        state_probs = {
            state['STATE']: float(pred_probs[idx]) 
            for idx, state in enumerate(priority_state)
        }
        
        predictions.append({
            'frame_idx': df.iloc[i + sequence_length - 1]['frame_idx'],
            'current_state': df.iloc[i + sequence_length - 1]['teddy_state'],
            'predicted_states': state_probs,
            'predicted_next_state': max(state_probs.items(), key=lambda x: x[1])[0]
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    output_path = annotation_df_path.replace('.csv', '_predictions.csv')
    predictions_df.to_csv(output_path, index=False)

    inference_times_df = pd.DataFrame({'frame_idx': predictions_df['frame_idx'], 'inference_time': inference_times})
    inference_times_df.to_csv(annotation_df_path.replace('.csv', '_inference_times.csv'), index=False)
    
    return predictions_df

if __name__ == '__main__':
    print("Inference ML: " + model_path)
    if len(sys.argv) < 2:
        print("Usage: python inference.py <video_path>")
        sys.exit(1)

    annotation_df_path = sys.argv[1]

    if not annotation_df_path.endswith('.csv'):
        print("Error: The provided file is not a CSV file.")
        sys.exit(1)

    generate_predictions(annotation_df_path, model_path, scaler_path)
