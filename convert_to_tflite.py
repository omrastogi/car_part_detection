import ai_edge_torch
import torch
import numpy
import torchvision
from src.model.convnext import build_convnext
import os
os.environ["PJRT_DEVICE"] = "CPU"

model = build_convnext(num_classes=7)
state_dict = torch.load("checkpoints/checkpoint_iter_5000.pth")
model.load_state_dict(state_dict["model_state_dict"])  # Replace with your saved model path
sample_inputs = (torch.randn(1, 3, 380, 380),)

torch_output = model(*sample_inputs)

# Convert PyTorch model to LiteRT format
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)

# Run inference on LiteRT model
edge_output = edge_model(*sample_inputs)

# Validate outputs
if numpy.allclose(
    torch_output.detach().numpy(),
    edge_output,
    atol=1e-5,
    rtol=1e-5,
):
    print("Inference results match between PyTorch and LiteRT!")
else:
    print("Discrepancy in results. Check your conversion.")

edge_model.export('model.tflite')
