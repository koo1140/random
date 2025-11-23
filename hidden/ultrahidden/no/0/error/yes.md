# here is the code. SCROLL DOWN TO UNDERSTAND

```py
# ---------------- simplicity-1a compatible base + fine-tune trainer ----------------
import os, re, json, time, math, glob
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from huggingface_hub import login

# ---------------- CONFIG ----------------
DO_TRAIN = True                # Set False to skip training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 20000
MAX_PAIRS = SAMPLE_SIZE*4
SEQ_LEN = 64
BATCH_SIZE = 128
EPOCHS = 3
EARLY_STOP = 3
D_MODEL = 512
NHEAD = 8
NLAYERS = 6
DROPOUT = 0.1
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_RESP_TOKENS = 20
CANDIDATES = 8
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY_BATCH = 5000
MAX_KEEP_CHECKPOINTS = 6
FINAL_CHECKPOINT_NAME = "final_model.pth"
BEST_CHECKPOINT_NAME = "best_model.pth"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------- helpers ----------------
def clean(s):
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()

def try_login(token):
    if not token:
        print("HF token not provided; skipping login.")
        return
    try:
        login(token)
        print("Logged in to Hugging Face (token cached).")
    except Exception as e:
        print("HF login warning:", e)

# ---------------- dataset streaming ----------------
def extract_pairs_from_example(example):
    pairs = []
    def _get_text(d):
        if isinstance(d, dict):
            return d.get("content") or d.get("text") or d.get("message")
        return None

    # lmsys style 'conversation'
    if "conversation" in example and isinstance(example["conversation"], list):
        conv = example["conversation"]
        for i in range(len(conv)-1):
            a,b = conv[i], conv[i+1]
            if isinstance(a, dict) and isinstance(b, dict):
                ra = str(a.get("role","")).lower()
                rb = str(b.get("role","")).lower()
                ta = _get_text(a)
                tb = _get_text(b)
                if isinstance(ta,str) and isinstance(tb,str) and ra.startswith("user") and rb.startswith("assistant"):
                    pairs.append((clean(ta), clean(tb)))

    # messages field
    if not pairs and "messages" in example:
        msgs = example["messages"]
        if isinstance(msgs,list) and len(msgs)>0:
            if isinstance(msgs[0], dict):
                for i in range(len(msgs)-1):
                    a,b = msgs[i], msgs[i+1]
                    ra = str(a.get("role","")).lower()
                    rb = str(b.get("role","")).lower()
                    ta = _get_text(a)
                    tb = _get_text(b)
                    if isinstance(ta,str) and isinstance(tb,str) and ra.startswith("user") and rb.startswith("assistant"):
                        pairs.append((ta,tb))
            elif isinstance(msgs[0], str):
                for i in range(0,len(msgs)-1,2):
                    if isinstance(msgs[i], str) and isinstance(msgs[i+1], str):
                        pairs.append((msgs[i], msgs[i+1]))

    # prompt/completion style
    if not pairs and "prompt" in example and "completion" in example:
        p,c = example["prompt"], example["completion"]
        if isinstance(p,str) and isinstance(c,str):
            pairs.append((p,c))

    return pairs

# ---------------- Dataset class ----------------
class StreamDataset(Dataset):
    def __init__(self, pairs, word2idx, seq_len):
        self.tokens = []
        self.seq_len = seq_len
        self.PAD_IDX = word2idx["<pad>"]
        for u,a in pairs:
            toks = (u + " <sep> " + a).split()
            self.tokens.extend([word2idx.get(t,self.PAD_IDX) for t in toks])
        self.N = max(0, len(self.tokens)-seq_len)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        seq = self.tokens[i:i+self.seq_len]
        tgt = self.tokens[i+self.seq_len]
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor(tgt, dtype=torch.long)
        return x,y

# ---------------- Model ----------------
class BigTinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS, dropout=DROPOUT, max_seq=SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model)
        self.pos = nn.Parameter(torch.zeros(1,max_seq,d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        B,T = x.size()
        h = self.embed(x) + self.pos[:,:T,:].to(x.device)
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        logits = self.fc(h)
        return logits

# ---------------- Checkpoints ----------------
def checkpoint_path(epoch, step):
    return os.path.join(CHECKPOINT_DIR,f"ckpt_ep{epoch:03d}_step{step:09d}.pth")

def save_checkpoint(model, optimizer, epoch, step, best_val):
    path = checkpoint_path(epoch, step)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val": best_val
    }, path)
    print("✅ Checkpoint saved:", path)
    return path

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        data = torch.load(path,map_location=DEVICE)
        model.load_state_dict(data["model_state"])
        optimizer.load_state_dict(data["optimizer_state"])
        return data.get("epoch",1), data.get("step",0), data.get("best_val",float("inf"))
    return 1,0,float("inf")

# ---------------- Training function (compatible) ----------------
def train_model(pairs, word2idx, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, epochs=EPOCHS, base_weights=None):
    vocab_size = len(word2idx)
    model = BigTinyTransformer(vocab_size, max_seq=seq_len).to(DEVICE)
    if base_weights:
        print("Loading base weights for fine-tuning...")
        model.load_state_dict(torch.load(base_weights,map_location=DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = StreamDataset(pairs, word2idx, seq_len)
    val_len = int(len(dataset)*0.05)
    train_len = len(dataset)-val_len
    train_ds, val_ds = random_split(dataset,[train_len,val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")

    global_step = 0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits,yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
            if global_step%100==0:
                print(f"Epoch {epoch} Step {global_step} Loss {loss.item():.4f}")
            if global_step%CHECKPOINT_EVERY_BATCH==0:
                save_checkpoint(model,optimizer,epoch,global_step,best_val)
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb),yb).item()
        val_loss/=len(val_loader)
        print(f"Epoch {epoch} done | Train {total_loss/len(train_loader):.4f} | Val {val_loss:.4f}")
        if val_loss<best_val:
            best_val = val_loss
            torch.save(model.state_dict(), BEST_CHECKPOINT_NAME)
            print("✅ New best model saved")
    torch.save(model.state_dict(), FINAL_CHECKPOINT_NAME)
    print("🎉 Training complete!")

# ---------------- Pipeline ----------------
def main(train_base=True, fine_tune=False, base_ckpt=None):
    # Stream dataset
    print("Streaming dataset...")
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    except:
        ds = load_dataset("daily_dialog", split="train", streaming=True)
    pairs=[]
    count=0
    for ex in ds:
        pairs.extend(extract_pairs_from_example(ex))
        count+=1
        if count>=SAMPLE_SIZE:
            break
    print(f"Extracted {len(pairs)} user->assistant pairs")
    # Build vocab
    corpus=[]
    for u,a in pairs:
        corpus.append(u+" <sep> "+a)
    tokens=set(" ".join(corpus).split())
    vocab=["<pad>","<sep>"] + sorted(list(tokens-{"<pad>","<sep>"}))
    word2idx={w:i for i,w in enumerate(vocab)}
    idx2word={i:w for i,w in enumerate(vocab)}
    with open("vocab.json","w") as f: json.dump(vocab,f)
    print("Vocab size:", len(vocab))
    # Train
    if train_base:
        print("Starting BASE model training...")
        train_model(pairs, word2idx)
    elif fine_tune and base_ckpt:
        print("Starting FINE-TUNE on base model...")
        train_model(pairs, word2idx, base_weights=base_ckpt)
    else:
        print("No training mode selected.")

# Example usage:
# To train base model from scratch:
# main(train_base=True)
# To fine-tune on new data:
# main(train_base=False, fine_tune=True, base_ckpt="final_model.pth")

```

# to understand:
>>>>>>>>> - does this code u wrote rn does both train and finetune in one workflow?
- No — it does either base training or fine-tuning, depending on the flags you pass to main(). It does not automatically do both in one run.

>>>>>>>>> - so it means i run it once for train and once for finetune
- Exactly — you run it once to train the base model, then run it again to fine-tune on new data.

>>>>>>>>> - this means that it also fully works like my old train with attention and advanced scoring system, all other stuff. and after fientune it has chat template
- NO. its not complete

>>>>>>>>> - what datset is used for what step
- Base training → TinyStories; fine-tuning → your chat dataset (e.g., LMSYS-chat or other user→assistant pairs).

hope its enough :P


# bellow is probably the full code- i dont know lmfao (train base + fientune + score functions and every other stuff) - its longer code so probably its full

```py
"""
single-file: base train -> fine-tune -> chat + rerank
Usage:
  python pipeline.py --mode train_base       # train base on TinyStories
  python pipeline.py --mode finetune         # fine-tune base on chat dataset
  python pipeline.py --mode chat             # run interactive chat (loads best checkpoint)
  python pipeline.py --mode all              # train base then finetune then chat
Notes:
  - Requires: pip install datasets transformers torch
  - Make sure HF token is available in env if you need gated datasets.
"""

import os, sys, argparse, math, time, json, random, re
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# model / training config (tweak as needed)
SEQ_LEN = 128
D_MODEL = 512
NHEAD = 8
NLAYERS = 6
DROPOUT = 0.1

BASE_BATCH = 48           # base training batch (TinyStories)
FT_BATCH = 64             # fine-tune batch (chat)
BASE_EPOCHS = 2
FT_EPOCHS = 3
LR = 5e-4
WEIGHT_DECAY = 0.01

# datasets / streaming limits
BASE_LIMIT = 50000      # how many TinyStories samples to stream (approx)
FT_LIMIT = 50000        # how many chat rows to stream for finetune

# checkpoints
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
BASE_CKPT = CHECKPOINT_DIR / "base_tinytales.pth"
FT_CKPT  = CHECKPOINT_DIR / "ft_chat.pth"
BEST_CKPT = CHECKPOINT_DIR / "best_model.pth"

# sampling / rerank settings defaults
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95
DEFAULT_CANDIDATES = 8
DEFAULT_MAX_RESP_TOKENS = 40

# separator token (just a string we include between user/assistant in training)
SEP = "<sep>"

# ---------------- tokenizer ----------------
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id
VOCAB_SIZE = len(tokenizer)
print("Vocab size (tokenizer):", VOCAB_SIZE)

# ---------------- Model ----------------
class TinyGPT(nn.Module):
    """Decoder-only model implemented with TransformerDecoder for simplicity."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS, seq_len=SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=4*d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.fc.weight = self.embed.weight

    def forward(self, x):
        """
        x: (B, T) token ids
        returns: (B, T, vocab)
        """
        B, T = x.size()
        if T > self.seq_len:
            raise ValueError(f"Sequence length {T} > model max {self.seq_len}")
        # causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        key_pad_mask = (x == PAD_ID)  # True where padding
        pos = self.pos_enc[:, :T, :].to(x.device)
        h = self.embed(x) * math.sqrt(self.d_model) + pos
        h = self.drop(h)
        # memory is the same sequence (decoder-only)
        h = self.decoder(h, memory=h, tgt_mask=mask, tgt_key_padding_mask=key_pad_mask, memory_mask=mask)
        h = self.norm(h)
        logits = self.fc(h)
        return logits

# ---------------- Datasets (streaming) ----------------
class TinyStoriesDataset(IterableDataset):
    """Streams TinyStories with GPT2 tokenizer"""
    def __init__(self, tokenizer, max_len=SEQ_LEN, limit=BASE_LIMIT):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.limit = limit
        self._ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def __iter__(self):
        count = 0
        for ex in self._ds:
            text = ex.get("text", "")
            if not isinstance(text, str): continue
            enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
            ids = enc["input_ids"][0]
            if ids.numel() < 8: continue
            yield ids
            count += 1
            if count >= self.limit: break

def extract_pairs_from_example(example):
    """Robust extractor (similar to your older script). Returns list of (user, assistant) strings."""
    pairs = []
    def _get_text(d):
        if isinstance(d, dict):
            return d.get("content") or d.get("text") or d.get("message")
        return None

    if "conversation" in example and isinstance(example["conversation"], list):
        conv = example["conversation"]
        for i in range(len(conv)-1):
            a,b = conv[i], conv[i+1]
            if isinstance(a, dict) and isinstance(b, dict):
                ra = str(a.get("role","")).lower()
                rb = str(b.get("role","")).lower()
                ta = _get_text(a); tb = _get_text(b)
                if isinstance(ta,str) and isinstance(tb,str) and ra.startswith("user") and rb.startswith("assistant"):
                    pairs.append((ta, tb))
    if not pairs and "messages" in example:
        msgs = example["messages"]
        if isinstance(msgs, list) and len(msgs) > 0:
            if isinstance(msgs[0], dict):
                for i in range(len(msgs)-1):
                    a,b = msgs[i], msgs[i+1]
                    ra = str(a.get("role","")).lower(); rb = str(b.get("role","")).lower()
                    ta = _get_text(a); tb = _get_text(b)
                    if isinstance(ta,str) and isinstance(tb,str) and ra.startswith("user") and rb.startswith("assistant"):
                        pairs.append((ta,tb))
            elif isinstance(msgs[0], str):
                for i in range(0, len(msgs)-1, 2):
                    if isinstance(msgs[i], str) and isinstance(msgs[i+1], str):
                        pairs.append((msgs[i], msgs[i+1]))
    if not pairs and "prompt" in example and "completion" in example:
        p,c = example["prompt"], example["completion"]
        if isinstance(p, str) and isinstance(c, str):
            pairs.append((p, c))
    # fallback
    return pairs

class ChatStreamDataset(IterableDataset):
    """Stream chat pairs and return tokenized sequences: user <sep> assistant"""
    def __init__(self, tokenizer, max_len=SEQ_LEN, limit=FT_LIMIT, preferred="lmsys/lmsys-chat-1m", fallback="daily_dialog"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.limit = limit
        try:
            ds = load_dataset(preferred, split="train", streaming=True)
            self._ds = ds
            print("Streaming preferred chat dataset:", preferred)
        except Exception as e:
            print("Preferred dataset failed:", e, "Falling back to", fallback)
            self._ds = load_dataset(fallback, split="train", streaming=True)

    def __iter__(self):
        count = 0
        for ex in self._ds:
            pairs = extract_pairs_from_example(ex)
            for u,a in pairs:
                text = (u.strip() + " " + SEP + " " + a.strip()).strip()
                if not text: continue
                enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
                ids = enc["input_ids"][0]
                if ids.numel() < 6: continue
                yield ids
                count += 1
                if count >= self.limit:
                    return

# collate
def collate_token_batches(batch):
    # batch: list of 1D tensors
    return pad_sequence(batch, batch_first=True, padding_value=PAD_ID)

# ---------------- Training utilities ----------------
def save_state(path, model, optimizer=None, epoch=None, step=None, extra=None):
    payload = {"model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if epoch is not None: payload["epoch"] = epoch
    if step is not None: payload["step"] = step
    if extra is not None: payload.update(extra)
    torch.save(payload, str(path))
    print("Saved checkpoint:", path)

def load_state(path, model, optimizer=None):
    if not os.path.exists(path):
        print("Checkpoint not found:", path)
        return None
    data = torch.load(path, map_location=DEVICE)
    if "model_state" in data:
        model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        try:
            optimizer.load_state_dict(data["optimizer_state"])
        except Exception as e:
            print("Warning: failed to load optimizer state:", e)
    return data

# ---------------- Sampling & Scoring (token-level) ----------------
def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """Apply top-k and/or top-p (nucleus) filtering to logits (1D tensor)."""
    logits = logits.clone()
    # top-k
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, top_k)
        min_topk = topk_vals[-1]
        logits[logits < min_topk] = filter_value
    # top-p
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # mask tokens where cumulative prob exceeds p
        sorted_indices_to_remove = cum_probs > top_p
        # keep at least first token
        if sorted_indices_to_remove[0]:
            sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def sample_autoregressive(model, tokenizer, prompt_ids, max_new_tokens=40, temperature=1.0, top_k=40, top_p=0.95, device=DEVICE):
    """
    Autoregressive sampling (token-level).
    prompt_ids: 1D tensor of token ids (already on device)
    returns (token_list, sum_logprob)
    """
    model.eval()
    x = prompt_ids.unsqueeze(0)  # (1, T)
    out_tokens = [int(t.item()) for t in x[0]]
    lp_sum = 0.0
    for _ in range(max_new_tokens):
        if len(out_tokens) >= SEQ_LEN:
            break
        inp = torch.tensor([out_tokens], dtype=torch.long, device=device)
        logits = model(inp)  # (1, seq, vocab)
        logits_step = logits[0, -1, :] / max(1e-6, temperature)
        filtered = _top_k_top_p_filtering(logits_step, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered, dim=-1)
        if torch.isnan(probs).any() or probs.sum().item() <= 0:
            # fallback to argmax
            next_id = int(torch.argmax(logits_step).item())
            p = probs[next_id].item() if probs.sum().item() > 0 else 1.0
        else:
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            p = probs[next_id].item()
        out_tokens.append(next_id)
        lp_sum += math.log(max(1e-12, p))
        # stop on EOS token (gpt2 eos is also pad) or SEP token id
        if next_id == tokenizer.eos_token_id:
            break
        # optional: break if SEP encountered at end (but keep it)
    return out_tokens, lp_sum

# scoring helpers (operate on decoded text / words)
def _clean_text_for_scoring(text):
    s = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return s

def topic_score(decoded_words, topic_tokens):
    if not topic_tokens: return 0.0
    cnt = 0
    for t in topic_tokens:
        cnt += sum(1 for w in decoded_words if w == t)
    return float(cnt)

def repetition_penalty(decoded_words):
    consec = 0
    for a,b in zip(decoded_words, decoded_words[1:]):
        if a == b: consec += 1
    uniq = len(set(decoded_words)) / max(1, len(decoded_words))
    return float(consec), float(uniq)

def score_candidate(decoded_text, logprob, topic_tokens=None, rep_weight=0.2, topic_weight=0.1):
    """
    Compute score:
      score = logprob + topic_weight*topic_score - rep_weight*consecutive_repeats
    decoded_text: string
    """
    if topic_tokens is None: topic_tokens = []
    cleaned = _clean_text_for_scoring(decoded_text)
    words = cleaned.split()
    tboost = topic_score(words, topic_tokens) * topic_weight
    consec, uniq = repetition_penalty(words)
    rep_pen = consec * rep_weight
    return logprob + tboost - rep_pen

# alias to match your earlier talked name
def score_candidates(decoded_text, logprob, topic_tokens=None, **kwargs):
    return score_candidate(decoded_text, logprob, topic_tokens=topic_tokens, **kwargs)

def generate_and_rerank(model, tokenizer, prompt_text, num_words=DEFAULT_MAX_RESP_TOKENS,
                        temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P,
                        candidates=DEFAULT_CANDIDATES, topic_tokens=None, device=DEVICE):
    """
    Generate multiple candidate continuations and rerank them.
    prompt_text: str (raw text)
    returns (best_text, best_score, best_tokens)
    """
    # tokenize prompt
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    prompt_ids = enc["input_ids"][0].to(device)
    cand_list = []
    for _ in range(candidates):
        toks, lp = sample_autoregressive(model, tokenizer, prompt_ids, max_new_tokens=num_words,
                                         temperature=temperature, top_k=top_k, top_p=top_p, device=device)
        text = tokenizer.decode(toks, skip_special_tokens=True)
        score = score_candidate(text, lp, topic_tokens=topic_tokens)
        cand_list.append((score, text, toks, lp))
    best = max(cand_list, key=lambda x: x[0])
    return best[1], best[0], best[2]

# ---------------- Training loops ----------------
def train_loop(model, dataloader, epochs, lr=LR, save_path=None, resume_from=None, tag="train"):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    start_epoch = 1
    global_step = 0
    best_val = float("inf")

    # resume
    if resume_from and os.path.exists(resume_from):
        print("Resuming from", resume_from)
        data = load_state(resume_from, model, optimizer)
        if data:
            start_epoch = int(data.get("epoch", 1))
            global_step = int(data.get("step", 0))
            best_val = float(data.get("best_val", float("inf")))

    for epoch in range(start_epoch, epochs+1):
        model.train()
        t0 = time.time()
        tot_loss = 0.0
        nb = 0
        for batch in dataloader:
            # batch is (B, T)
            batch = batch.to(DEVICE)
            x = batch[:, :-1]
            y = batch[:, 1:]
            if x.size(1) == 0: continue
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x)                # (B, T, V)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item()
            nb += 1
            global_step += 1
            if global_step % 100 == 0:
                print(f"[{tag}] Epoch {epoch}/{epochs} step {global_step} loss {loss.item():.4f}")
        avg_loss = tot_loss / max(1, nb)
        t1 = time.time()
        print(f"[{tag}] Epoch {epoch} completed | avg loss {avg_loss:.4f} | time {(t1-t0):.1f}s")
        # save checkpoint
        if save_path is not None:
            extra = {"epoch": epoch, "step": global_step, "best_val": best_val}
            save_state(save_path, model, optimizer, epoch=epoch, step=global_step, extra=extra)
    return model

# ---------------- High level actions ----------------
def train_base(base_epochs=BASE_EPOCHS, batch=BASE_BATCH, resume_ckpt=None):
    print("Training BASE model on TinyStories...")
    ds = TinyStoriesDataset(tokenizer, max_len=SEQ_LEN, limit=BASE_LIMIT)
    loader = DataLoader(ds, batch_size=batch, collate_fn=collate_token_batches)
    model = TinyGPT(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS, seq_len=SEQ_LEN, dropout=DROPOUT)
    train_loop(model, loader, epochs=base_epochs, lr=LR, save_path=BASE_CKPT, resume_from=resume_ckpt, tag="base")
    print("Base training done. Saved to", BASE_CKPT)

def finetune_on_chat(ft_epochs=FT_EPOCHS, batch=FT_BATCH, base_ckpt=None, resume_ckpt=None):
    print("Fine-tuning on chat dataset...")
    ds = ChatStreamDataset(tokenizer, max_len=SEQ_LEN, limit=FT_LIMIT)
    loader = DataLoader(ds, batch_size=batch, collate_fn=collate_token_batches)
    model = TinyGPT(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS, seq_len=SEQ_LEN, dropout=DROPOUT)
    # load base if provided
    if base_ckpt and os.path.exists(base_ckpt):
        print("Loading base checkpoint:", base_ckpt)
        load_state(base_ckpt, model, None)
    train_loop(model, loader, epochs=ft_epochs, lr=LR, save_path=FT_CKPT, resume_from=resume_ckpt, tag="finetune")
    print("Fine-tune done. Saved to", FT_CKPT)

def interactive_chat(model_ckpt=None):
    print("Starting interactive chat. Loading model checkpoint...")
    model = TinyGPT(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS, seq_len=SEQ_LEN, dropout=DROPOUT)
    if model_ckpt and os.path.exists(model_ckpt):
        load_state(model_ckpt, model, None)
        print("Loaded model from", model_ckpt)
    else:
        # try FT then base
        if FT_CKPT.exists():
            load_state(str(FT_CKPT), model, None); print("Loaded ft checkpoint")
        elif BASE_CKPT.exists():
            load_state(str(BASE_CKPT), model, None); print("Loaded base checkpoint")
        else:
            print("No checkpoint found. You must train or provide a checkpoint.")
            return
    model.to(DEVICE); model.eval()

    # chat variables
    temperature = DEFAULT_TEMPERATURE
    top_k = DEFAULT_TOP_K
    top_p = DEFAULT_TOP_P
    candidates = DEFAULT_CANDIDATES
    topic_tokens = []

    print("\n💬 Chat ready. Type /help, 'quit' to exit.")
    print("Commands: /temp <val> /topk <n> /topp <p> /cands <n> /topic <words> /reset /help")
    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        if not user:
            continue
        if user.lower() == "quit":
            break
        if user.startswith("/help"):
            print("Commands: /temp <val>, /topk <n>, /topp <p>, /cands <n>, /topic <words>, /reset, quit")
            continue
        if user.startswith("/temp"):
            try: temperature = float(user.split()[1]); print("✅ temperature", temperature)
            except: print("usage /temp 1.0"); continue
        if user.startswith("/topk"):
            try: top_k = int(user.split()[1]); print("✅ top_k", top_k)
            except: print("usage /topk 40"); continue
        if user.startswith("/topp"):
            try: top_p = float(user.split()[1]); print("✅ top_p", top_p)
            except: print("usage /topp 0.95"); continue
        if user.startswith("/cands"):
            try: candidates = int(user.split()[1]); print("✅ candidates", candidates)
            except: print("usage /cands 8"); continue
        if user.startswith("/topic"):
            rest = user[len("/topic"):].strip()
            if not rest:
                topic_tokens = []
                print("✅ topic cleared")
            else:
                toks = re.sub(r"[^\w\s]", "", rest.lower()).split()
                topic_tokens = toks
                print("✅ topic set:", topic_tokens)
            continue
        if user.startswith("/reset"):
            history = []; print("✅ memory cleared"); continue

        # prepare prompt: include history + user + SEP
        # For simplicity we use only the latest user -> prompt, but you can append previous turns.
        prompt = user + " " + SEP + " "
        # generate and rerank
        try:
            best_text, best_score, _ = generate_and_rerank(model, tokenizer, prompt,
                                                           num_words=DEFAULT_MAX_RESP_TOKENS,
                                                           temperature=temperature, top_k=top_k, top_p=top_p,
                                                           candidates=candidates, topic_tokens=topic_tokens, device=DEVICE)
            # Basic cleaning: remove leading prompt repetition if model echoes prompt
            # If model includes prompt in output, show only continuation after prompt length
            # decode prompt length tokens to detect
            print("AI:", best_text)
        except Exception as e:
            print("Generation error:", e)

# ---------------- CLI entry ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train_base", "finetune", "chat", "all"], default="chat")
    p.add_argument("--resume", type=str, default=None, help="path to resume checkpoint")
    p.add_argument("--base_ckpt", type=str, default=str(BASE_CKPT), help="path to base checkpoint to load for finetune")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mode = args.mode

    if mode == "train_base":
        train_base()
    elif mode == "finetune":
        finetune_on_chat(base_ckpt=args.base_ckpt, resume_ckpt=args.resume)
    elif mode == "chat":
        interactive_chat(model_ckpt=args.resume if args.resume else None)
    elif mode == "all":
        # train base then finetune then chat
        train_base()
        finetune_on_chat(base_ckpt=str(BASE_CKPT))
        interactive_chat(model_ckpt=str(FT_CKPT))
    else:
        print("Unknown mode:", mode)


```


# you probably scrolled too down, look above and u will find Q&A.
