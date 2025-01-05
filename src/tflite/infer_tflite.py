import tensorflow as tf
import numpy as np
from PIL import Image
import autoroot
import autorootcwd

class TFLiteModel:
    def __init__(self, model_path):
        """
        Initialize the TFLite model.
        
        Args:
            model_path (str): Path to the TFLite model file.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Extract input shape
        self.input_shape = self.input_details[0]['shape']  # Typically (1, 3, height, width)
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

    def preprocess_image(self, image_path):
        """
        Preprocess the input image for the model.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            np.ndarray: Preprocessed image tensor.
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.input_width, self.input_height))
        image_array = np.asarray(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image_array = np.transpose(image_array, (2, 0, 1))  # Convert to (3, height, width)
        return np.expand_dims(image_array, axis=0)  # Add batch dimension

    def predict(self, image_path):
        """
        Predict the output for a given input image.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            int: Predicted class index.
        """
        input_tensor = self.preprocess_image(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        return predicted_class

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Infer a TFLite model on a given image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the TFLite model file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    model = model = TFLiteModel(args.model_path)
    prediction = model.predict(args.image_path)
    print("Predicted Class:", prediction)
