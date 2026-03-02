
import torch
import os

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    try:
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = torch.matmul(a, b)
        print("CUDA Matmul Successful")
    except Exception as e:
        print(f"CUDA Matmul Failed: {e}")
else:
    print("CUDA not available.")
