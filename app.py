import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = os.urandom(24)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

with open("data/junpei_v3_vocab.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

vocab_size = len(stoi)
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

batch_size = 1
block_size = 128
n_embd = 128
n_layer = 4
n_head = 4
dropout = 0.2

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        logits = logits.reshape(B*T, vocab_size)
        targets = targets.reshape(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

model = GPT().to(device)
weights_path = "data/junpei_v3_weights.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print(f"✅ Model loaded from {weights_path}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("prompt", "").strip()
    if not user_message:
        return jsonify({"response": "Please type something."})

    if "history" not in session:
        session["history"] = []
    history = session["history"][-5:]
    history.append({"role": "user", "content": user_message})

    prompt_text = "User and Junpei-LLM are chatting.\n"
    for msg in history:
        role = "User" if msg["role"] == "user" else "Junpei"
        prompt_text += f"{role}: {msg['content']}\n"
    prompt_text += "Junpei: " 
    context_tensor = torch.tensor(
        encode(prompt_text), dtype=torch.long, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        response_idx = model.generate(
            context_tensor, max_new_tokens=80, temperature=0.7
        )

    raw_output = decode(response_idx[0].tolist())

    if "Junpei:" in raw_output:
        raw_output = raw_output.split("Junpei:", 1)[-1].strip()

    if "User:" in raw_output:
        raw_output = raw_output.split("User:")[0].strip()

    response_text = raw_output.replace("\n", " ").strip()

    history.append({"role": "bot", "content": response_text})
    session["history"] = history[-5:]

    return jsonify({"response": response_text})



@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return jsonify({"response": "Chat history reset."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
