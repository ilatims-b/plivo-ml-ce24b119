"""BPE tokenizer trained on the provided train_corpus.txt only.
Lossless arbitrary UTF-8 byte-level fallback.
Exposes: load() -> tokenizer object with .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
"""
import json
import os
import re
from collections import Counter


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


class BPETokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.merges = []            # list of [p0, p1]
        self.merges_dict = {}       # (p0, p1) -> new_id
        self.merges_rank = {}       # (p0, p1) -> rank (0, 1, ...)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.cache = {}
        self._pattern = re.compile(r'\S+|\s+')

    def train(self, text, vocab_size=2048):
        self.vocab_size = vocab_size
        num_merges = vocab_size - 256

        words = self._pattern.findall(text)
        word_counts = Counter(tuple(w.encode("utf-8")) for w in words)

        merges = []
        vocab = {i: bytes([i]) for i in range(256)}

        pair_counts = Counter()
        pair_to_words = {}
        for w_tuple, count in word_counts.items():
            if len(w_tuple) < 2:
                continue
            for i in range(len(w_tuple) - 1):
                pair = (w_tuple[i], w_tuple[i + 1])
                pair_counts[pair] += count
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(w_tuple)

        for _ in range(num_merges):
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            if best_count <= 0:
                break

            new_id = 256 + len(merges)
            merges.append(list(best_pair))
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            del pair_counts[best_pair]

            affected_words = list(pair_to_words.pop(best_pair, []))
            for w_tuple in affected_words:
                if w_tuple not in word_counts:
                    continue
                count = word_counts.pop(w_tuple)

                for j in range(len(w_tuple) - 1):
                    p = (w_tuple[j], w_tuple[j + 1])
                    if p == best_pair:
                        continue
                    pair_counts[p] -= count
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    if p in pair_to_words and w_tuple in pair_to_words[p]:
                        pair_to_words[p].remove(w_tuple)

                new_w = []
                j = 0
                while j < len(w_tuple):
                    if j < len(w_tuple) - 1 and (w_tuple[j], w_tuple[j + 1]) == best_pair:
                        new_w.append(new_id)
                        j += 2
                    else:
                        new_w.append(w_tuple[j])
                        j += 1
                new_w_tuple = tuple(new_w)
                word_counts[new_w_tuple] = count

                if len(new_w_tuple) >= 2:
                    for j in range(len(new_w_tuple) - 1):
                        p = (new_w_tuple[j], new_w_tuple[j + 1])
                        pair_counts[p] = pair_counts.get(p, 0) + count
                        if p not in pair_to_words:
                            pair_to_words[p] = set()
                        pair_to_words[p].add(new_w_tuple)

        self.merges = merges
        self.vocab_size = 256 + len(merges)
        self.merges_dict = {tuple(m): 256 + i for i, m in enumerate(merges)}
        self.merges_rank = {tuple(m): i for i, m in enumerate(merges)}
        self.vocab = vocab
        self.cache = {}

    def encode(self, text):
        if not text:
            return []
        ids = []
        for word in self._pattern.findall(text):
            wb = word.encode("utf-8")
            if wb not in self.cache:
                w_ids = list(wb)
                while len(w_ids) >= 2:
                    best_idx = -1
                    best_rank = float("inf")
                    for i in range(len(w_ids) - 1):
                        pair = (w_ids[i], w_ids[i + 1])
                        rank = self.merges_rank.get(pair, float("inf"))
                        if rank < best_rank:
                            best_rank = rank
                            best_idx = i
                    if best_idx == -1:
                        break
                    pair = (w_ids[best_idx], w_ids[best_idx + 1])
                    new_id = self.merges_dict[pair]
                    new_w_ids = []
                    i = 0
                    while i < len(w_ids):
                        if i < len(w_ids) - 1 and (w_ids[i], w_ids[i + 1]) == pair:
                            new_w_ids.append(new_id)
                            i += 2
                        else:
                            new_w_ids.append(w_ids[i])
                            i += 1
                    w_ids = new_w_ids
                self.cache[wb] = w_ids
            ids.extend(self.cache[wb])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save_file(self, path):
        data = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_file(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("type") != "bpe":
            return
        self.merges = data["merges"]
        self.vocab_size = 256 + len(self.merges)
        self.merges_dict = {tuple(m): 256 + i for i, m in enumerate(self.merges)}
        self.merges_rank = {tuple(m): i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, m in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[m[0]] + self.vocab[m[1]]
        self.cache = {}


def load(path=None):
    """Return the tokenizer used by evaluate.py and train.py."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "bpe_vocab.json")
    if os.path.exists(path):
        tok = BPETokenizer()
        tok.load_file(path)
        return tok
    train_path = os.path.join(os.path.dirname(__file__), "../data/train_corpus.txt")
    if os.path.exists(train_path):
        tok = BPETokenizer()
        text = open(train_path, encoding="utf-8").read()
        tok.train(text, vocab_size=2048)
        tok.save_file(path)
        return tok
    return ByteTokenizer()
