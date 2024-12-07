import tensorflow as tf
from .model import WeldingDetectionModel
from ..data.data_processing import DataProcessor
import os

class ModelTrainer:
    def __init__(self):
        self.model = WeldingDetectionModel()
        self.data_processor = DataProcessor()
        self.batch_size = 32
        self.epochs = 50
        
    def train(self):
        """Training process"""
        # Prepare data
        X, y = self.data_processor.prepare_dataset()
        
        # Split train/validation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Build and train model
        model = self.model.build_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                'data/models/best_model.h5',
                save_best_only=True
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        return history