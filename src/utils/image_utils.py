import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.min_area = 10
        self.max_area = 200
        
    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {path}")
        return img
        
    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges
        
    def detect_welding_points(self, image):
        edges = self.preprocess(image)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        return points