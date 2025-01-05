import os
import argparse
import pandas as pd
from src.tflite.infer_tflite import TFLiteModel  # Import your TFLiteModel class

# Class labels
class_labels = ["Front", "Rear", "Front-Right", "Front-Left", "Rear-Right", "Rear-Left", "None"]

def main(args):
    # Initialize the model
    model = TFLiteModel(args.model)

    # Prepare results list
    results = []

    # Iterate through images in the folder
    for file_name in os.listdir(args.folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(args.folder, file_name)
            
            # Predict the class
            predicted_index = model.predict(file_path)
            predicted_class = class_labels[predicted_index]
            
            # Append results
            results.append({"image_name": file_name, "prediction": predicted_class})
    
    # Create a DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(args.csv, index=False)
    print(f"Predictions saved to {args.csv}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Test TFLite Model on Images")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--model", type=str, required=True, help="Path to the TFLite model file")
    parser.add_argument("--csv", type=str, default="predictions.csv", help="Output CSV file name (default: predictions.csv)")
    
    args = parser.parse_args()
    main(args)
