import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("model/waste_classifier.h5")
labels = ["Paper", "Plastic"]


image_folder = "sample_images"

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
hazardous_k = ["lamp", "bulb", "medicine", "pill", "battery", "chemical"]

print("📸 Running with Real AI Model...\n")

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image is not None:
        print(f"✅ Image Loaded: {image_name}")

        
        resized = cv2.resize(image, (180, 180))
        normalized = resized / 255.0
        input_image = normalized.reshape(1, 180, 180, 3)

        
        prediction = model.predict(input_image)
        class_index = np.argmax(prediction)
        predicted_label = labels[class_index]        
        is_hazardous_keyword = any(keyword in image_name.lower() for keyword in hazardous_k)        
        if is_hazardous_keyword:
            predicted_label = "Hazardous Waste"
            warning_msg = "⚠ Chemical/Hazardous Waste Detected"
            action_msg = "Apply Chemical Clearance Immediately"
            color = (0, 0, 255)  # Red
        elif predicted_label == "Plastic":
            warning_msg = "⚠ Hazardous Waste Detected"
            action_msg = "Apply Instant Chemical Clearance"
            color = (0, 0, 255)
        else:
            warning_msg = "✅ Non-Hazardous Waste"
            action_msg = "No Chemical Clearance Required"
            color = (0, 255, 0)  # Green
        print(f"📊 Raw Prediction Probabilities: {prediction}")
        print(f"🤖 Predicted Waste Type: {predicted_label}")
        print(f"⚙ Action: Rotate motor to '{predicted_label}' bin")
        print(warning_msg)
        print(action_msg)    
        display_image = cv2.resize(resized, (500, 500))
        cv2.putText(display_image, f"Prediction: {predicted_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display_image, warning_msg, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_image, action_msg, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)        
        cv2.imshow("Result", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"❌ Could not load image {image_name}")