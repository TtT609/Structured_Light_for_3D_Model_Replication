#include <Stepper.h>

// 17500 steps for a full 360 degree rotation
const int stepsPerRevolution = 17500; 



// Initialize pins: IN1, IN3, IN2, IN4 (Specific order for ULN2003)
Stepper myStepper(stepsPerRevolution, 19, 5, 18, 17);

void setup() {
  myStepper.setSpeed(10); // 10 RPM is stable for this motor
  
  // Start Serial communication at the exact baud rate expected by your Python script
  Serial.begin(115200); 
  Serial.println("Motor initialized. Waiting for Python commands...");
}

void loop() {
  // Check if the Python script sent a command over PySerial
  if (Serial.available() > 0) {
    
    // Read the incoming text from Python (e.g., "30.0")
    String input = Serial.readStringUntil('\n');
    input.trim(); // Remove any extra spaces or newline characters
    
    // Convert the string into a decimal number
    float degrees = input.toFloat();
    
    // If the angle is valid, execute the movement
    if (degrees > 0) {
      // Calculate how many steps the motor needs to move for that angle
      long stepsToMove = (degrees / 360.0) * stepsPerRevolution;
      
      Serial.print("Moving ");
      Serial.print(degrees);
      Serial.println(" degrees...");
      
      // Move the motor (this blocks the code until the movement is finished)
      myStepper.step(stepsToMove);
      
      // Tell the Python script that the rotation is complete so it can take the picture!
      Serial.println("DONE");
    }
  }
}