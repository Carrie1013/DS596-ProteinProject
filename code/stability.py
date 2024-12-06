import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import load_pretrained_model
import matplotlib.pyplot as plt

class Stability:
    def __init__(self, benchmark_name='stability', benchmarks_dir='../data/benchmark'):
        self.benchmark_name = benchmark_name
        self.benchmarks_dir = benchmarks_dir
        self.pretrained_model_generator, self.input_encoder = load_pretrained_model()
        self.model = None

    def load_data(self):
        train_set_file_path = os.path.join(self.benchmarks_dir, f'{self.benchmark_name}.train.csv')
        test_set_file_path = os.path.join(self.benchmarks_dir, f'{self.benchmark_name}.test.csv')

        train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
        train_set, valid_set = train_test_split(train_set, test_size=0.1, random_state=0)

        test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

    def encode_sequences(self, sequences, seq_len=512):
        encoded = self.input_encoder.encode_X(sequences, seq_len=seq_len)
        return encoded[0] if isinstance(encoded, list) else encoded

    def prepare_features(self):
        self.train_features = self.encode_sequences(self.train_set['seq'], seq_len=512)
        self.valid_features = self.encode_sequences(self.valid_set['seq'], seq_len=512)
        self.test_features = self.encode_sequences(self.test_set['seq'], seq_len=512)

    def build_model(self, input_dim, output_dim=1):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(output_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        self.model = model

    def train(self, epochs=10, batch_size=32):
        if self.model is None:
            self.build_model(input_dim=self.train_features.shape[1])

        history = self.model.fit(
            self.train_features, self.train_set['label'],
            validation_data=(self.valid_features, self.valid_set['label']),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
            # callbacks=[
            #     keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            # ]
        ).history

        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Loss', color='blue')
        plt.plot(history['val_loss'], label='Validation Loss', color='red')
        plt.title('Loss Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self):
        test_loss, test_mse = self.model.evaluate(self.test_features, self.test_set['label'], batch_size=32)
        print(f'Test Loss (MSE): {test_loss}, Test MSE: {test_mse}')
        return test_loss, test_mse

    def predict(self, sequences):
        sequences_encoded = self.encode_sequences(sequences, seq_len=512)
        predictions = self.model.predict(sequences_encoded, verbose=0)
        return predictions
    

    def predict_stabilities(self, data):

        left_stabilities = []
        right_stabilities = []
        
        for _, row in data.iterrows():

            split_site = row['Split Site']
            sequence = row['Sequence']
            
            left_seq = sequence[:split_site]
            right_seq = sequence[split_site:]
            
            left_stability = self.predict([left_seq])[0][0]
            right_stability = self.predict([right_seq])[0][0]
            
            left_stabilities.append(left_stability)
            right_stabilities.append(right_stability)
        
        data['Left Stability'] = left_stabilities
        data['Right Stability'] = right_stabilities
        data['Stability Sum'] = data['Left Stability'] + data['Right Stability']
        
        return data

# Example Usage:
# stability_model = Stability()
# stability_model.load_data()
# stability_model.prepare_features()
# stability_model.train()
# stability_model.evaluate()
# predictions = stability_model.predict(["SEQUENCE_HERE"])
# print(predictions)
