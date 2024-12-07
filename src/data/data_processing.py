import cv2
import numpy as np
import os
import json

class DataProcessor:
    def __init__(self):
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"
        self.labels_file = "data/labels.json"
        
    def preprocess_image(self, image):
        """Tiền xử lý ảnh"""
        # Resize
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
        
    def augment_data(self, image):
        """Data augmentation"""
        augmented = []
        
        # Flip horizontal
        augmented.append(cv2.flip(image, 1))
        
        # Rotate
        angles = [90, 180, 270]
        for angle in angles:
            matrix = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (224, 224))
            augmented.append(rotated)
            
        return augmented
        
    def prepare_dataset(self):
        """Chuẩn bị dataset cho training"""
        images = []
        labels = []
        
        with open(self.labels_file, 'r') as f:
            label_data = json.load(f)
            
        for image_name, points in label_data.items():
            image_path = os.path.join(self.raw_dir, image_name)
            image = cv2.imread(image_path)
            
            if image is not None:
                processed = self.preprocess_image(image)
                images.append(processed)
                labels.append(points)
                
                # Thêm augmented data
                augmented = self.augment_data(processed)
                images.extend(augmented)
                labels.extend([points] * len(augmented))
                
        return np.array(images), np.array(labels)