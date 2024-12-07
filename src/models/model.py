import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

class WeldingDetectionModel:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model = None
        
    def build_model(self):
        """Xây dựng model architecture"""
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(2, activation='sigmoid')  # x, y coordinates
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
        
    def save_model(self, path):
        """Lưu model"""
        self.model.save(path)
        
    def load_model(self, path):
        """Load model"""
        self.model = tf.keras.models.load_model(path)