"""
bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
sequences of integers, where each integer represents small chunks of commonly
occuring characters. This implementation is based on openai's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
but was mildly modified because the original implementation is a bit confusing.
I also tried to add as many comments as possible, my own understanding of what's
going on.
"""

import os
import json
import regex as re
import requests

import torch

# -----------------------------------------------------------------------------

def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually. Some bytes have their appearance preserved
    because they don't cause any trouble. These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
    bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). Instead,
    this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
    that "look nice", either in their original form, or a funny shifted character
    like 'Ā', or 'Ġ', etc.
    """
    # the 188 integers that render fine in their original form and need no shifting
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] # all integers b in bs will simply map to chr(b) in the output dict
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    n = 0
    for b in range(2**8):
        if b not in bs:
            # if this byte is "ugly" then map it to the next available "nice" character
            bs.append(b)
            cs.append(2**8+n) # appending higher characters to prevent any kind of overlap with english letters in data
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs)) # used dict so why even care of order 
    return d

def get_pairs(word):
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word.
    """
    pairs = set() # no duplicates
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges):
        # byte encoder/decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        # bpe token encoder/decoder
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        # Use <|endoftext|> as EOS
        self.eos_token = '<|endoftext|>'
        self.pad_token = '<|pad|>'
        max_id = max(self.encoder.values())
        if self.pad_token not in self.encoder:
            self.encoder[self.pad_token] = max_id + 1
            self.decoder[max_id + 1] = self.pad_token
        # bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # the splitting pattern used for pre-tokenization
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions <-- original openai comment
        """
        ok so what is this regex looking for, exactly?
        python re reference: https://docs.python.org/3/library/re.html
        - the vertical bars | is OR, so re.findall will chunkate text as the pieces match, from left to right
        - '\'s' would split up things like Andrej's -> (Andrej, 's)
        - ' ?\p{L}': optional space followed by 1+ unicode code points in the category "letter"
        - ' ?\p{N}': optional space followed by 1+ unicode code points in the category "number"
        - ' ?[^\s\p{L}\p{N}]+': optional space, then 1+ things that are NOT a whitespace, letter or number
        - '\s+(?!\S)': 1+ whitespace characters (e.g. space or tab or etc) UNLESS they are followed by non-whitespace
                       so this will consume whitespace characters in a sequence but exclude the last whitespace in
                       that sequence. that last whitespace has the opportunity to then match the optional ' ?' in
                       earlier patterns.
        - '\s+': 1+ whitespace characters, intended probably to catch a full trailing sequence of whitespaces at end of string
        So TLDR:
        - we are special casing a few common apostrophe constructs ('s, 't, 're, ...) and making those into separate tokens
        - we then separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces
        """
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        """
        this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens
        up the tree. token is a string of one individual 'word' (after regex tokenization)
        and after byte encoding, e.g. 'Ġthere'.
        """
        # token is a string of one individual 'word', after byte encoding, e.g. 'Ġthere'

        # memoization, for efficiency
        if token in self.cache:  # if already processed this token
            return self.cache[token]

        word = tuple(token) # individual characters that make up the token, in a tuple
        pairs = get_pairs(word) # get all bigrams

        if not pairs: # if word has only 1 character
            return token

        while True:

            # find the next lowest rank bigram that can be merged
            # If the pair is not found, it assigns it rank ∞ (so it’s never chosen)
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:  # if no valid pair exists
                break # no more bigrams are eligible to be merged
            first, second = bigram

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            new_word = []
            i = 0
            while i < len(word):

                # find the next occurence of first in the sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # concat all words into a string, and use ' ' as the separator. Note that
        # by now all characters have been byte encoded, guaranteeing that ' ' is
        # not used in the actual data and is a 'special' delimiter character
        word = ' '.join(word)

        # cache the result and return
        self.cache[token] = word
        return word

    def encode(self, text, add_eos=False):
        """ string goes in, list of integers comes out """
        bpe_idx = []
        # pre-tokenize the input text into string tokens (words, roughly speaking)
        tokens = re.findall(self.pat, text)
        # process each token into BPE integers
        for token in tokens:
            # encode the token as a bytes (b'') object
            token_bytes = token.encode('utf-8') #  GPT tokenizer starts from bytes, not characters
            # translate all bytes to their unicode string representation and flatten
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            # perform all the applicable bpe merges according to self.bpe_ranks
            token_merged = self.bpe(token_translated).split(' ')
            # translate all bpe tokens to integers
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged] # that's the gpt2 feature used and so was the ranking of bigrams or else how would it be possbile to encode
            # extend our running list of all output integers
            bpe_idx.extend(token_ix) # continues for each token in the loop
        if add_eos:
            bpe_idx.append(self.encoder[self.eos_token])
        return bpe_idx # token ids of entire input


    """                                             :SOME NOTES:
    Yes — **exactly right**. You just nailed a core idea of how GPT-2's tokenizer *must* work.

Let me break your insight down clearly so it locks in:

---

### 🔁 Why BPE Merging **Must Follow GPT-2's Pretrained Ranking**

#### 🧠 What GPT-2 expects:

* GPT-2 was trained with a very specific vocabulary of **50,257 tokens**, defined in:

  * `encoder.json`: maps token strings (like `'Ġhello'`) → token IDs (integers)
  * `vocab.bpe`: tells how to merge characters into those token strings

Those token strings are **subwords**, not full words. For example:

* `'Ġun'`, `'believ'`, `'able'` — three BPE tokens that make up `'Ġunbelievable'`

---

### 🚫 Why You Can’t Just Use Your Own Ranking

If you change the BPE merge ranking (e.g. count bigrams from your own input and rank them), you’d:

1. Merge tokens **differently** from how GPT-2 was trained
2. Generate subword tokens that **don’t exist** in `encoder.json`
3. 💥 Get a `KeyError` when calling:

   ```python
   self.encoder[bpe_token]
   ```

Because GPT-2 literally doesn’t know what that made-up token is.

---

### ✅ Why GPT-2’s Pretrained Merge Rules Matter

Using the same `vocab.bpe` ensures:

* All merged subword tokens match GPT-2’s known vocabulary
* You can map each token → integer with `encoder.json`
* You can feed those integers into GPT-2’s model weights (which expect those same tokens)

---

### 📌 Think of It Like a Compiler

* `encode()` is like compiling text to bytecode
* GPT-2 is trained to read *one specific bytecode format* (its 50k-token vocab)
* If you mess with the compiler logic (your own merge ranks), GPT-2 can’t interpret the result

---

### 💡 Analogy

Imagine GPT-2 as a brain that only understands **its own subword language**.
If you invent your own grammar and try to speak to it, it’ll just stare back at you blankly — or crash.

---

So yes — you were spot on.
The **ranking** ensures that BPE merges happen in a way that aligns with the **token IDs GPT-2 understands** — allowing `encode()` to work and the model to function.

Great questions. You're now touching the **core mechanics** of BPE — the merge ranking logic and how tokens are replaced. Let's tackle both parts **cleanly and deeply**:

---

## ❓ 1. **How is the ranking of each BPE pair decided? Is it random?**

**Not random. Not arbitrary. It's entirely frequency-based.**

### 🔢 Ranking is learned during *offline training* on a large corpus:

1. Start with every word split into characters (e.g., `h e l l o`)
2. Count how often **each bigram (pair)** occurs across the whole training corpus.
3. Pick the **most frequent pair**, e.g., `('l', 'l')`, and merge it → becomes `'ll'`
4. Update the corpus by replacing every `('l', 'l')` with `'ll'`
5. Repeat until you have the desired number of merges (like 50k)

Each merge is recorded in order:

```python
[('l', 'l'), ('ll', 'o'), ('e', 'l'), ...]
```

That ordered list becomes `vocab.bpe`, and this part of the code assigns a **ranking**:

```python
self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
```
    """
    def encode_and_show_work(self, text):
        """ debugging function, same as encode but returns all intermediate work """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ') # returns a merged string where ' ' is the delimiter between tokens. so used split(' ') at the end
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token, # raw token
                'token_bytes': token_bytes, # bytes version
                'token_translated': token_translated, # translated Unicode string
                'token_merged': token_merged, # list of bpe subwords after merging
                'token_ix': token_ix, # token IDs
            })
        out = {
            'bpe_idx': bpe_idx, # the actual output sequence
            'tokens': tokens, # result of pre-tokenization
            'parts': parts, # intermediates for each token(pre-tokenized regex applied) part
        }
        return out

    def decode(self, bpe_idx):
        """ list of integers comes in, string comes out """
        out_tokens = []
        for token in bpe_idx:
            if token == self.encoder.get('<|endoftext|>'):
                break
            if token == self.encoder.get('<|pad|>'):
                continue
            out_tokens.append(token)
        # inverse map the integers to get the tokens
        tokens_merged = [self.decoder[token] for token in out_tokens]
        # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes
        tokens_flat = ''.join(tokens_merged)
        # Each character in tokens_flat is mapped back to its original byte
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # recover the full utf-8 string
        text = tokens_bytes.decode('utf-8', errors='replace') # this method is of bytearray class
        return text

def get_file(local_file, remote_file):
    """ downloads remote_file to local_file if necessary """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)

def get_encoder():
    """
    Returns an instance of the GPT BPE Encoder/Decoder
    and handles caching of "database" files.
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    # load encoder.json that has the raw mappings from token -> bpe index
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == 50257 # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token

    # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure
    # in the form tuples (a, b), that indicate that (a, b) is to be merged to one token ab
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # light postprocessing: strip the version on first line and the last line is a blank
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000 # 50,000 merged tokens

    # construct the Encoder object and return
    enc = Encoder(encoder, bpe_merges)
    return enc

# -----------------------------------------------------------------------------

# class used to make above code compatible with pytorch
class BPETokenizer:
    """ PyTorch-aware class that wraps the Encoder above """

    def __init__(self):
        self.encoder = get_encoder()
        self.eos_token = '<|endoftext|>'
        self.pad_token = '<|pad|>'
        self.eos_token_id = self.encoder.encoder[self.eos_token]
        self.pad_token_id = self.encoder.encoder[self.pad_token]

    def __call__(self, text, return_tensors='pt'): #  makes the object behave like a function
        # PyTorch only; here because we want to match huggingface/transformers interface
        assert return_tensors == 'pt'
        # single string input for now, in the future potentially a list of strings
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1
        idx = [self.encoder.encode(text)]
        # wrap into PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long) # [batch_size, seq_len]
        return out

    def decode(self, idx):
        # ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # decode indices to text
        text = self.encoder.decode(idx.tolist()) # tensors back to list
        return text


if __name__ == '__main__': # run this file directly to see how it works and not as a module

    # here is an encoding example
    text = "Hello!! I'm Naman Chanana. It's 2025. These wordings are currently in bpe.py file. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("Original text is:")
    print(text)
    print("First the text gets pre-tokenized, broken up into chunks, the outcome is:")
    print(r['tokens'])
    # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    print("Then we iterate over each chunk and process them in turn...")
    for part in r['parts']:
        print(part)
    # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    # (refer to the code inside Encoder.encode for what these intermediates are)
    print("and the final outcome is concatenating and flattening all the token_ix:")
    print(r['bpe_idx'])
    # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # this would then become the integer input sequence to the transformer
    print("ready to feed into a Transformer!")
