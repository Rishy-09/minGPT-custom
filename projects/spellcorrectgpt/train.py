import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset
import types

# ✅ Add custom minGPT path
sys.path.append("/kaggle/input/spellcorrectgpt-code/minGPT-custom")

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

set_seed(42)

# ✅ Configs
block_size = 128
tokenizer = BPETokenizer()
pad_token_id = tokenizer.pad_token_id
vocab_size = len(tokenizer.encoder.encoder)

# ✅ Dataset class using (noisy, clean) from CSV
class SpellCorrectDataset(Dataset):
    def __init__(self, dataframe, tokenizer, block_size):
        self.data = list(zip(dataframe['augmented_text'], dataframe['text']))
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy, clean = self.data[idx]
        noisy_ids = self.tokenizer.encoder.encode(noisy)
        sep_ids = self.tokenizer.encoder.encode("=>")
        clean_ids = self.tokenizer.encoder.encode(clean, add_eos=True)

        input_ids = noisy_ids + sep_ids + clean_ids
        x = input_ids[:self.block_size]

        target_ids = [-100] * (len(noisy_ids) + len(sep_ids)) + clean_ids
        y = target_ids[:self.block_size]

        pad_len = self.block_size - len(x)
        if pad_len > 0:
            x += [self.pad_token_id] * pad_len
            y += [-100] * pad_len

        for i, token in enumerate(y):
            if token != -100 and (token < 0 or token >= vocab_size):
                print(f"[❌ Bad Target] idx={idx}, token={token}, position={i}")
                print(f"  ➪ Noisy: {noisy}")
                print(f"  ➪ Clean: {clean}")
                raise ValueError("Invalid token ID in target detected.")

        return torch.tensor(x), torch.tensor(y)

# ✅ Load train & val datasets
base_path = "/kaggle/input/spelling-mistake-data-1mn"
train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
val_df = pd.read_csv(os.path.join(base_path, "val.csv"))

train_dataset = SpellCorrectDataset(train_df, tokenizer, block_size)
val_dataset = SpellCorrectDataset(val_df, tokenizer, block_size)

print(f"\n📁 Dataset Loaded | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ✅ Model Config
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-mini'
model_config.vocab_size = vocab_size  # ← ensure matches tokenizer
model_config.block_size = block_size

model = GPT(model_config)

# ✅ Load pretrained weights before training
ckpt_path = "/kaggle/input/spellcorrectgpt-code/minGPT-custom/spellcorrectgpt_final.pt"
model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))  # or 'cuda' if you're sure
print(f"✅ Loaded pretrained weights from {ckpt_path}")


# ✅ Patch model to ignore padding loss
def patched_forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
    x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    loss = None
    if targets is not None:
        # ✅ Assert all target tokens are either -100 or valid vocab IDs
        assert torch.all((targets == -100) | ((targets >= 0) & (targets < logits.size(-1)))), \
            f"Invalid target token found. Max target: {targets.max()}, Vocab size: {logits.size(-1)}"

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
    return logits, loss

model.forward = types.MethodType(patched_forward, model)

# ✅ Trainer Setup
trainer_config = Trainer.get_default_config()
trainer_config.max_iters = 22000
trainer_config.eval_interval = 1000
trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(trainer_config, model, train_dataset)

def decode_and_show(model, x, tokenizer, device, label=None):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x.to(device))
        preds = torch.argmax(logits, dim=-1)

        for i in range(min(3, x.size(0))):  # Show max 3 samples per batch
            input_text = tokenizer.encoder.decode([t for t in x[i].tolist() if t != pad_token_id])
            pred_text = tokenizer.encoder.decode([
                t for t, y_t in zip(preds[i].tolist(), x[i].tolist())
                if y_t != pad_token_id
            ])
            print(f"\n📝 Input:  {input_text}")
            print(f"🔮 Pred:   {pred_text}")
            if label is not None:
                true_text = tokenizer.encoder.decode([
                    t for t in label[i].tolist() if t != -100
                ])
                print(f"✅ Target: {true_text}")


# ✅ Evaluation utilities
def evaluate_val_loss(model, val_dataset, batch_size=32):
    model.eval()
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(trainer.device), y.to(trainer.device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("nan")

# ✅ Add logging hook
def on_batch_end(trainer):
    if trainer.pbar:
        trainer.pbar.update(1)
    if trainer.iter_num == 1 or trainer.iter_num % trainer.config.eval_interval == 0:
        print(f"\n⚙️ Iteration {trainer.iter_num} | Train Loss: {trainer.loss.item():.4f}")
        val_loss = evaluate_val_loss(trainer.model, val_dataset)
        print(f"📉 Val Loss: {val_loss:.4f}")
        

trainer.add_callback("on_batch_end", on_batch_end)

# ✅ Start Training
print(f"\n🚦 Starting Training | Device: {trainer_config.device}")
print(f"🕗 Total Steps: {trainer_config.max_iters} | Eval Interval: {trainer_config.eval_interval}")

try:
    if tqdm is not None:
        with tqdm(total=trainer_config.max_iters, desc="🚀 Training", ncols=100) as pbar:
            trainer.pbar = pbar
            trainer.run()
            trainer.pbar = None
    else:
        trainer.run()
except RuntimeError as e:
    print("\n❌ Training crashed due to RuntimeError.")
    print(f"🧠 Error Message: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

# ✅ Save model
final_ckpt_path = "/kaggle/working/spellcorrectgpt_final.pt"
torch.save(model.state_dict(), final_ckpt_path)
print(f"\n📄 Final model saved to: {final_ckpt_path}")
