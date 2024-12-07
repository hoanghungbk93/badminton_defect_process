import cv2
import numpy as np
import os
import json

class DataCollector:
    def __init__(self):
        self.data_dir = "data/raw"
        self.processed_dir = "data/processed"
        self.labels_file = "data/labels.json"
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def collect_from_camera(self, num_samples=100):
        """Thu thập dữ liệu từ camera"""
        cap = cv2.VideoCapture(0)
        
        count = 0
        while count < num_samples:
            ret, frame = cap.read()
            if ret:
                filename = f"{self.data_dir}/sample_{count}.jpg"
                cv2.imwrite(filename, frame)
                count += 1
                
        cap.release()
        
    def label_data(self, image_path, points):
        """Gán nhãn cho ảnh"""
        labels = {}
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                labels = json.load(f)
                
        image_name = os.path.basename(image_path)
        labels[image_name] = points
        
        with open(self.labels_file, 'w') as f:
            json.dump(labels, f)