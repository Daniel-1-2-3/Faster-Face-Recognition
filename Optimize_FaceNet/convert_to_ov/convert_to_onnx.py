import torch
from facenet_pytorch import InceptionResnetV1

torch_model = InceptionResnetV1(pretrained='vggface2').eval()
torch_model.load_state_dict(torch.load('C:\\Daniel\\Python\\OnClick Face Recognition\\Optimize_FaceNet\\model_versions\\quant_torch_model.pth'), strict=False)
example_input = torch.randn(1, 1, 299, 299)
onnx_program = torch.onnx.dynamo_export(torch_model, example_input)
onnx_program.save('onnx_quant_model.onnx')