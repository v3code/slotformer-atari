import torch
import torch.nn as nn

print(torch.load("slate_navigation5x5.pth")["ocr_module_state_dict"].keys())