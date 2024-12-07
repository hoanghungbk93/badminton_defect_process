from Jetson import GPIO
import time

class MotorController:
    def __init__(self):
        # Setup GPIO pins
        self.STEP_PIN_X = 18
        self.DIR_PIN_X = 23
        self.STEP_PIN_Y = 24
        self.DIR_PIN_Y = 25
        self.ENABLE_PIN = 17
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.STEP_PIN_X, self.DIR_PIN_X,
                   self.STEP_PIN_Y, self.DIR_PIN_Y,
                   self.ENABLE_PIN], GPIO.OUT)
                   
    def move_to(self, x, y):
        # Convert coordinates to steps
        steps_x = self.coord_to_steps(x)
        steps_y = self.coord_to_steps(y)
        
        # Move motors
        self.move_motor(self.STEP_PIN_X, self.DIR_PIN_X, steps_x)
        self.move_motor(self.STEP_PIN_Y, self.DIR_PIN_Y, steps_y)
        
    def cleanup(self):
        GPIO.cleanup()