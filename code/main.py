import os
import shap
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import load_pretrained_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class Split:
    def __init__(self, filepath, max_len=128, seq_len=1500, pca_components=80):
        
        self.filepath = filepath
        self.max_len = max_len
        self.seq_len = seq_len
        self.pca_components = pca_components
        self.sampled_df = None
        self.secondary_encoded = None
        self.sequence_encoded = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_reduced = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_components)

    def load_data(self):
        # self.sampled_df = pd.read_csv(self.filepath, index_col="Unnamed: 0")
        self.sampled_df = pd.read_csv(self.filepath)

    def encode_secondary_structures_alternative(self):
        secondary_structures = self.sampled_df['Secondary']
        unique_chars = sorted(list(set("".join(secondary_structures))))
        char_to_index = {char: i for i, char in enumerate(unique_chars)}
        num_unique_chars = len(unique_chars)

        sequences_numeric = [
            [char_to_index[char] for char in seq[:self.max_len]] + [0] * (self.max_len - len(seq[:self.max_len]))
            for seq in secondary_structures
        ]
        
        one_hot_encoded = np.zeros((len(secondary_structures), self.max_len, num_unique_chars))
        for i, seq in enumerate(sequences_numeric):
            for j, idx in enumerate(seq):
                one_hot_encoded[i, j, idx] = 1

        self.secondary_encoded = one_hot_encoded

    def encode_sequences_with_merge(self):
        pretrained_model_generator, input_encoder = load_pretrained_model()
        sequences = self.sampled_df['Sequence']
        encoded = input_encoder.encode_X(sequences, seq_len=self.seq_len)
        self.sequence_encoded = encoded[0] if isinstance(encoded, list) else encoded

    def combine_features(self):
        secondary_flattened = self.secondary_encoded.reshape(self.secondary_encoded.shape[0], -1)
        self.X = np.hstack([
            np.array(self.sequence_encoded),
            secondary_flattened,
            self.sampled_df[['Site']].values,
            self.sampled_df[['Stability Sum']].values
        ])
        self.y = self.sampled_df['Split'].astype(int).values

    def scale_features(self):
        self.X_scaled = self.scaler.fit_transform(self.X)

    def apply_pca(self):
        self.X_reduced = self.pca.fit_transform(self.X_scaled)

    def train_mlp(self, test_size=0.2, random_state=42, hidden_layer_sizes=(128, 64), max_iter=500):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_reduced, self.y, test_size=test_size, random_state=random_state
        )
        self.mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
        self.mlp_model.fit(X_train, y_train)

        y_pred = self.mlp_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def process(self):
        self.load_data()
        self.encode_secondary_structures_alternative()
        self.encode_sequences_with_merge()
        self.combine_features()
        self.scale_features()
        self.apply_pca()

    def eval(self, df_test):
        
        required_columns = ['Site', 'Sequence', 'Secondary', 'Stability Sum']
        # required_columns = ['Site', 'Sequence', 'Secondary']
        for col in required_columns:
            if col not in df_test.columns:
                raise KeyError(f"Test dataset must contain the column '{col}'.")

        secondary_structures = df_test['Secondary']
        unique_chars = sorted(list(set("".join(self.sampled_df['Secondary']))))  # 使用训练集的字符集
        char_to_index = {char: i for i, char in enumerate(unique_chars)}
        num_unique_chars = len(unique_chars)

        sequences_numeric = [
            [char_to_index.get(char, 0) for char in seq[:self.max_len]] + [0] * (self.max_len - len(seq[:self.max_len]))
            for seq in secondary_structures
        ]
        secondary_encoded_test = np.zeros((len(secondary_structures), self.max_len, num_unique_chars))
        for i, seq in enumerate(sequences_numeric):
            for j, idx in enumerate(seq):
                secondary_encoded_test[i, j, idx] = 1
        secondary_flattened_test = secondary_encoded_test.reshape(secondary_encoded_test.shape[0], -1)

        pretrained_model_generator, input_encoder = load_pretrained_model()
        sequences = df_test['Sequence']
        sequence_encoded_test = input_encoder.encode_X(sequences, seq_len=self.seq_len)
        sequence_encoded_test = sequence_encoded_test[0] if isinstance(sequence_encoded_test, list) else sequence_encoded_test

        X_test = np.hstack([
            np.array(sequence_encoded_test),
            secondary_flattened_test,
            df_test[['Site']].values,
            df_test[['Stability Sum']].values
        ])

        X_test_scaled = self.scaler.transform(X_test)
        X_test_reduced = self.pca.transform(X_test_scaled)
        y_test_pred = self.mlp_model.predict(X_test_reduced)

        return X_test_reduced, y_test_pred
    
    def test_eval(self, df_test, y_test):

        y_test_pred = self.eval(df_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred)

        return accuracy, report


# Example usage:
# split_processor = Split(filepath="unsampled.csv")
# split_processor.process()
