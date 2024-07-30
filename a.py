import torch
import torch.nn.functional as F

pis = torch.zeros(7, 15)

print("------------- 1")
print(pis, pis.size())

try:
    probs = F.softmax(pis, dim=0)
    print("------------- 2")
    print(probs)
except Exception as e:
    print(f"An error occurred: {e}")
