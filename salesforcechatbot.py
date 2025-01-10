import gptmodel as gpt
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mload = gpt.GPTLanguageModel()
mload.load_state_dict(torch.load("trainedmodel.pt", weights_only=True))
mload.to(device)
mload.eval()
print("Model Loaded")

inputstring = input("\nEnter a question for salesforce chatbot: ")
inputcontext = torch.tensor([gpt.encode(inputstring)], device=device)
print("Answer:\n")
print(gpt.decode(mload.generate(inputcontext, max_new_tokens=250)[0].tolist()))