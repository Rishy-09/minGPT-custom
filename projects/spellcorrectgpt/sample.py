# sample.py

import os
import sys
import torch
import pandas as pd
import types
from jiwer import wer, cer
import numpy as np
import nltk

nltk.download('punkt')

# ✅ Custom minGPT path
sys.path.append("/kaggle/input/spellcorrectgpt-code/minGPT-custom")

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer

tokenizer = BPETokenizer()
block_size = 128
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

# ✅ Load model
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-mini'
model_config.vocab_size = len(tokenizer.encoder.encoder)
model_config.block_size = block_size

model = GPT(model_config)

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
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
    return logits, loss

model.forward = types.MethodType(patched_forward, model)

# ✅ Load weights
ckpt_path = "/kaggle/input/spellcorrectgpt-code/minGPT-custom/spellcorrectgpt_final.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
model.eval()

# ✅ Load test data
test_path = "/kaggle/input/spelling-mistake-data-1mn/test.csv"
df = pd.read_csv(test_path)
assert 'augmented_text' in df.columns and 'text' in df.columns

MAX_SAMPLES = 15
print(f"\n🔍 Running inference on {MAX_SAMPLES} test samples...\n")

predictions = []

for i, noisy in enumerate(df['augmented_text'].iloc[:MAX_SAMPLES]):
    if pd.isna(noisy) or not noisy.strip():
        predictions.append("")
        continue

    prompt = f"{noisy.strip()} =>"
    input_ids = tokenizer.encoder.encode(prompt, add_eos=False)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=64,
            temperature=0.3,
            top_k=10,
            do_sample=True,
            
        )

    # ✅ Extract only the predicted portion after "=>"
    decoded = tokenizer.encoder.decode(y[0].tolist())
    if "=>" in decoded:
        prediction = decoded.split("=>", 1)[1].strip()
    else:
        prediction = decoded.strip()

    # ✅ Clean hallucinated double sentences if any
    prediction = prediction.split(tokenizer.encoder.eos_token, 1)[0].strip()
    predictions.append(prediction)

    target = df['text'].iloc[i].strip() if pd.notna(df['text'].iloc[i]) else ""
    print(f"📝 [{i+1}] Noisy     : {noisy.strip()}")
    print(f"🔮 [{i+1}] Predicted : {prediction}")
    print(f"✅ [{i+1}] Target    : {target}")
    print("-" * 50)

# ✅ Save predictions
df = df.iloc[:MAX_SAMPLES].copy()
df['predicted_text'] = predictions

output_path = "/kaggle/working/test_predictions.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Predictions saved to: {output_path}")

# ✅ Evaluation
targets = df['text'].fillna("").tolist()
preds = df['predicted_text'].fillna("").tolist()

exact_matches = [p.strip() == t.strip() for p, t in zip(preds, targets)]
exact_match_acc = sum(exact_matches) / len(exact_matches)

edit_distances = [nltk.edit_distance(p, t) for p, t in zip(preds, targets)]
avg_edit_distance = np.mean(edit_distances)

wer_score = wer(targets, preds)
cer_score = cer(targets, preds)

print("\n📊 Final Evaluation Metrics:")
print(f"✅ Exact Match Rate     : {exact_match_acc * 100:.2f}%")
print(f"✏️  Avg Edit Distance    : {avg_edit_distance:.2f}")
print(f"📉 Word Error Rate (WER): {wer_score * 100:.2f}%")
print(f"🔤 Char Error Rate (CER): {cer_score * 100:.2f}%")
