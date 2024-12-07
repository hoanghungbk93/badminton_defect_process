import unittest
import cv2
import numpy as np
from src.utils.image_utils import ImageProcessor
from src.models.model import WeldingDetectionModel

class TestWeldingDetection(unittest.TestCase):
    def setUp(self):
        self.image_processor = ImageProcessor()
        self.model = WeldingDetectionModel()
        
    def test_image_loading(self):
        """Test image loading"""
        image = self.image_processor.load_image('data/raw/test.jpg')
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)
        
    def test_preprocessing(self):
        """Test preprocessing"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = self.image_processor.preprocess(image)
        self.assertIsNotNone(processed)
        
    def test_point_detection(self):
        """Test welding point detection"""
        image = cv2.imread('data/raw/test.jpg')
        points = self.image_processor.detect_welding_points(image)
        self.assertIsInstance(points, list)
        
    def test_model_prediction(self):
        """Test model prediction"""
        self.model.load_model('data/models/best_model.h5')
        image = np.random.random((1, 224, 224, 3))
        prediction = self.model.model.predict(image)
        self.assertEqual(prediction.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()