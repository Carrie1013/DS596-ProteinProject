import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import load_pretrained_model


BENCHMARK_NAME = 'fluorescence'
BENCHMARKS_DIR = 'protein_bert/protein_benchmarks'

train_set_file_path = os.path.join(BENCHMARKS_DIR, f'{BENCHMARK_NAME}.train.csv')
train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
train_set, valid_set = train_test_split(train_set, test_size=0.1, random_state=0)

test_set_file_path = os.path.join(BENCHMARKS_DIR, f'{BENCHMARK_NAME}.test.csv')
test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()


# load the model_encoder
pretrained_model_generator, input_encoder = load_pretrained_model()

def encode_sequences_with_merge(sequences, encoder, seq_len=512):
    encoded = encoder.encode_X(sequences, seq_len=seq_len)
    valid_encoded = encoded[0] if isinstance(encoded, list) else encoded
    return valid_encoded

train_features = encode_sequences_with_merge(train_set['seq'], input_encoder, seq_len=512)
valid_features = encode_sequences_with_merge(valid_set['seq'], input_encoder, seq_len=512)
test_features = encode_sequences_with_merge(test_set['seq'], input_encoder, seq_len=512)


# MLP
def build_mlp(input_dim, output_dim=1):

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_dim, activation='linear')  # Regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return model

mlp_model = build_mlp(input_dim=train_features.shape[1])

# train the MLP
mlp_model.fit(
    train_features, train_set['label'],
    validation_data=(valid_features, valid_set['label']),
    epochs=20,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)


test_loss, test_mse = mlp_model.evaluate(test_features, test_set['label'], batch_size=32)
print(f'Test Loss (MSE): {test_loss}, Test MSE: {test_mse}')