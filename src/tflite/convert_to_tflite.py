import autoroot
import autorootcwd
import argparse
import ai_edge_torch
import torch
import numpy
import torchvision
from src.model.convnext import build_convnext
import os

def main(args):
    # Set the PJRT_DEVICE environment variable
    os.environ["PJRT_DEVICE"] = "CPU"

    # Load the model
    model = build_convnext(num_classes=args.num_classes)
    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict["model_state_dict"])  # Replace with your saved model path
    sample_inputs = (torch.randn(1, 3, args.input_size, args.input_size),)

    # Run PyTorch inference for validation
    torch_output = model(*sample_inputs)

    # Convert PyTorch model to LiteRT format
    edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)

    # Run inference on LiteRT model
    edge_output = edge_model(*sample_inputs)

    # Validate outputs
    if numpy.allclose(
        torch_output.detach().numpy(),
        edge_output,
        atol=args.atol,
        rtol=args.rtol,
    ):
        print("Inference results match between PyTorch and LiteRT!")
    else:
        print("Discrepancy in results. Check your conversion.")

    # Export the model to TFLite
    edge_model.export(args.output_path)
    print(f"Exported TFLite model to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite using AI Edge Torch.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the PyTorch checkpoint file.")
    parser.add_argument("--output_path", type=str, default="model.tflite", help="Path to save the exported TFLite model.")
    parser.add_argument("--input_size", type=int, default=380, help="Input image size (height and width).")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes for the model.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for output validation.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for output validation.")
    
    args = parser.parse_args()
    main(args)
