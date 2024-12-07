from utils.image_utils import ImageProcessor
from utils.gpio_utils import MotorController
import cv2
import time

def main():
    try:
        # Initialize components
        image_processor = ImageProcessor()
        motor_controller = MotorController()
        
        # Load and process image
        image = image_processor.load_image('data/raw/vot.jpg')
        points = image_processor.detect_welding_points(image)
        
        # Process each welding point
        for point in points:
            # Move to position
            motor_controller.move_to(point[0], point[1])
            
            # Wait for movement to complete
            time.sleep(1)
            
            # Perform grinding
            # Add grinding logic here
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        motor_controller.cleanup()

if __name__ == "__main__":
    main()