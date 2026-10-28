"""
Ex. No: 4 - LANGUAGE MODELING USING RNN
"""
import glob
import math
import os
import random
import string
import time
import unicodedata
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 1. Download & Extract Dataset if Missing
data_dir = "data/names"
if not os.path.exists(data_dir) or len(glob.glob(f"{data_dir}/*.txt")) == 0:
    print("Downloading dataset from PyTorch...")
    url = "https://download.pytorch.org/tutorial/data.zip"
    zip_path = "data.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
        print("Dataset extracted successfully.")
    except Exception as e:
        print(f"Could not download ({e}). Creating local sample names data...")
        os.makedirs(data_dir, exist_ok=True)
        sample_dict = {
            "Russian": ["Romanov", "Ulyanov", "Smirnov", "Ivanov", "Petrov"],
            "German": ["Gerber", "Engel", "Richter", "Mueller", "Schmidt"],
            "Spanish": ["Salazar", "Perez", "Alvarez", "Garcia", "Rodriguez"],
            "Chinese": ["Chen", "Huang", "In", "Wang", "Zhang"]
        }
        for cat, nms in sample_dict.items():
            with open(f"{data_dir}/{cat}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(nms))

# 2. Vocabulary & Helpers
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker

def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" and c in all_letters)

def readLines(filename):
    with open(filename, encoding="utf-8") as f:
        return [unicodeToAscii(line.strip()) for line in f if line.strip()]

category_lines = {}
all_categories = []
for filename in glob.glob(f"{data_dir}/*.txt"):
    cat = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(cat)
    category_lines[cat] = readLines(filename)

n_categories = len(all_categories)
print(f"Loaded {n_categories} categories: {all_categories[:5]}...")

# 3. Model Architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.dropout(self.o2o(output_combined))
        return self.softmax(output), hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# 4. Helpers for Tensor Vectorization
def categoryTensor(category):
    tensor = torch.zeros(1, n_categories)
    tensor[0][all_categories.index(category)] = 1.0
    return tensor

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1.0
    return tensor

def targetTensor(line):
    indexes = [all_letters.find(line[i]) for i in range(1, len(line))]
    indexes.append(n_letters - 1)
    return torch.LongTensor(indexes)

def randomTrainingExample():
    cat = random.choice(all_categories)
    line = random.choice(category_lines[cat])
    return categoryTensor(cat), inputTensor(line), targetTensor(line)

# 5. Training
rnn = RNN(n_letters, 128, n_letters)
criterion = nn.NLLLoss()
learning_rate = 0.0005

def train(cat_tensor, inp_tensor, tgt_tensor):
    tgt_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = torch.tensor(0.0)
    for i in range(inp_tensor.size(0)):
        output, hidden = rnn(cat_tensor, inp_tensor[i], hidden)
        loss = loss + criterion(output, tgt_tensor[i])
    loss.backward()
    for p in rnn.parameters():
        if p.grad is not None:
            p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item() / inp_tensor.size(0)

n_iters = 50000
print_every, plot_every = 5000, 500
all_losses, total_loss = [], 0
start = time.time()

print("\nStarting Training...")
for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss
    if iter % print_every == 0:
        elapsed = int(time.time() - start)
        print(f"{elapsed // 60}m {elapsed % 60}s ({iter} {iter / n_iters * 100:.0f}%) Loss: {loss:.4f}")
    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

# Plot loss
plt.figure(figsize=(7, 4))
plt.plot(all_losses, color="tab:blue")
plt.xlabel("Iterations (in hundreds)")
plt.ylabel("NLL Loss")
plt.title("Character RNN Training Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 6. Sampling
def sample(category, start_letter="A", max_length=20):
    with torch.no_grad():
        cat_tensor = categoryTensor(category)
        inp = inputTensor(start_letter)
        hidden = rnn.initHidden()
        output_name = start_letter
        for _ in range(max_length):
            output, hidden = rnn(cat_tensor, inp[0], hidden)
            topi = output.topk(1)[1][0][0].item()
            if topi == n_letters - 1:
                break
            output_name += all_letters[topi]
            inp = inputTensor(all_letters[topi])
        return output_name

def samples(category, start_letters="ABC"):
    print(f"\n--- Generated {category} Names ---")
    for ch in start_letters:
        print(f"Start '{ch}': {sample(category, ch)}")

for cat, letters in [("Russian", "RUS"), ("German", "GER"), ("Spanish", "SPA"), ("Chinese", "CHI")]:
    if cat in all_categories:
        samples(cat, letters)

