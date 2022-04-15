import torch
import torch.onnx

from model import get_model
from settings import keypoints_classes_ids2names, weights_path

def convert_onnx():
    with torch.no_grad():
        model = get_model(num_keypoints = len(keypoints_classes_ids2names), weights_path=weights_path)
        model.eval()

        # Let's create a dummy input tensor  
        dummy_input = torch.randn(1, 3, 416, 416)

        # Export the model   
        torch.onnx.export(model,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            "./out/lead_sectional_image.onnx",       # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput'], # the model's output names 
            dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                    'modelOutput' : {0 : 'batch_size'}}) 
        print(" ") 
        print('Model has been converted to ONNX')

if __name__ == "__main__":
    convert_onnx()