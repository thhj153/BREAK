import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("CUDA version (compiled):", torch.version.cuda)
print("CUDA device:", torch.cuda.get_device_name(0))
print("CUDA device capability:", torch.cuda.get_device_capability(0)) 
 