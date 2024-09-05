"""
Convert to ONNX format for deployment on raspberry pi/jetson nano
"""

# Import necessary libraries
import torch
import torch.onnx
from vitcnn import SmartBinSegregationModel

# Define the function to export the model to ONNX format
def export_model_to_onnx():
    # Initialize the SmartBinSegregationModel
    model = SmartBinSegregationModel()

    # Load the trained model weights if you have a saved model
    # Example: model.load_state_dict(torch.load('path_to_trained_model.pth'))
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input to trace the model
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust size based on your input format

    # Define the export file path
    onnx_file_path = "smartbin_segmentation_model.onnx"

    # Export the model to ONNX format
    torch.onnx.export(
        model,                              # Model instance
        dummy_input,                        # Dummy input for tracing
        onnx_file_path,                     # File path to save the ONNX model
        export_params=True,                 # Store the trained parameter weights inside the model file
        opset_version=11,                   # ONNX version
        do_constant_folding=True,           # Constant folding for optimization
        input_names=["input"],              # Input tensor names (for the input layer)
        output_names=["output"],            # Output tensor names (for the output layer)
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Allow variable batch sizes
    )

    print(f"Model has been converted to ONNX and saved at: {onnx_file_path}")

if __name__ == "__main__":
    export_model_to_onnx()
