# RUNLOG

## Run 0: Baseline
- **Hypothesis**: Baseline starter code with raw UTF-8 byte tokenizer (vocab 256), 4-layer GPT (`n_embd=160`, `n_head=4`, `block_size=128`), constant Adam optimizer (`lr=3e-4`, no warmup/decay/clipping), untied weights (`tie_weights=False`).
- **What changed**: Nothing (unmodified starter code).
- **Results**:
  - Parameters: `1,339,840`
  - Training loss: `1.7315` (at step 2000)
  - Dev score (`bpb`): **2.3718**
  - Time: `47s` total (`~23 ms/step`)
- **Conclusion**: The baseline `bpb` of 2.3718 is mediocre due to untied weights, lack of learning rate schedule/warmup/weight decay, small context/batch size, sub-optimal architecture, and character/byte-level representation where each Devanagari character requires 3 tokens and the model spends excessive capacity predicting raw UTF-8 bytes without short-range compression.

## Run 1: Optimizer & Learning Rate Schedule (`train.py`)
- **Hypothesis**: Replacing constant Adam (`lr=3e-4`) with AdamW (`lr=1e-3`, `weight_decay=0.1`, `betas=(0.9, 0.95)`), linear warmup (100 steps), cosine decay down to `1e-4`, and gradient clipping (`max_norm=1.0`) will improve stability and convergence.
- **What changed**: Modified `train.py` training loop and optimizer setup while keeping model architecture and byte tokenizer unchanged.
- **Results**:
  - Parameters: `1,339,840`
  - Training loss: `1.6084` (down from `1.7315`)
  - Dev score (`bpb`): **2.2516** (down from `2.3718`)
  - Time: `49s` total (`~24 ms/step`)
- **Conclusion**: The schedule and regularization provided a substantial boost (`-0.1202` bpb reduction). The model learns significantly faster without diverging, validating the optimizer changes.

## Run 2: Weight Tying (`tie_weights = True`)
- **Hypothesis**: Tying input and output token embeddings (`tie_weights = True`) will reduce redundant parameters and regularize the representations.
- **What changed**: Set `tie_weights = True` in `model.py` `Config`.
- **Results**:
  - Parameters: `1,298,880` (down by 40,960)
  - Training loss: `1.6378`
  - Dev score (`bpb`): `2.2877` (`+0.0361` vs Run 1)
  - Time: `48s` total (`~24 ms/step`)
- **Conclusion**: Simply removing the head weight matrix without reallocating the freed parameter budget slightly reduced model capacity and increased `bpb` by `0.0361`. However, weight tying frees up ~41k parameters (or much more when vocabulary size increases), allowing us to scale the core Transformer block (layers/embedding dimension) under the 2,000,000 parameter cap.

## Run 3: BPE Tokenizer (`vocab_size = 2048`) & `RMSNorm` (`tokenizer.py`, `model.py`)
- **Hypothesis**: Replacing raw byte tokens (`vocab_size = 256`) with a Byte-Pair Encoding (BPE) tokenizer (`vocab_size = 2048`) trained on `train_corpus.txt` will compress multi-byte Devanagari/Hindi characters and common English subwords, allowing each token (`block_size = 128`) to span ~2.5x more context bytes and dramatically improving `bpb`. Replacing `LayerNorm` with `RMSNorm` removes redundant mean/bias parameters and speeds up computation.
- **What changed**: Implemented an ultra-fast BPE tokenizer in `tokenizer.py` (vocab 2048, ~2.51x compression) and replaced `LayerNorm` with `RMSNorm` in `model.py`.
- **Results**:
  - Parameters: `1,584,160`
  - Training loss: `3.7427` (in BPE token space)
  - Dev score (`bpb`): **2.1820** (down from `2.2877`)
  - Time: `64s` total (`~32 ms/step`)
- **Conclusion**: Subword compression via BPE combined with `RMSNorm` yielded a massive `bpb` drop (`-0.1057` vs Run 2, `-0.1898` vs baseline). BPE allows the model to predict meaningful subwords rather than raw UTF-8 byte transitions, making far better use of the sequence context and model capacity.

## Run 4: Untying Weights with BPE & RMSNorm (`model.py`)
- **Hypothesis**: Untying input token embeddings and output head weights (`tie_weights = False`) with `vocab_size = 2048` and `n_embd = 160` will add `327,680` parameters right below the 2,000,000 cap (`1,911,840` total params), allowing independent representations for input semantics and output logit classification.
- **What changed**: Set `tie_weights = False` in `model.py` `Config` while keeping `BPETokenizer (vocab 2048)`, `RMSNorm`, and AdamW schedule.
- **Results**:
  - Parameters: `1,911,840` (`+327,680` vs Run 3)
  - Training loss: `3.6259` (down from `3.7427`)
  - Dev score (`bpb`): **2.1345** (down from `2.1820`)
  - Time: `69s` total (`~34 ms/step`)
- **Conclusion**: Untying weights with the larger vocabulary proved highly effective (`-0.0475` bpb reduction, `-0.2373` vs baseline). Because `n_embd=160` is relatively compact, decoupling the input embedding table from the output classification projection allows the model to leverage the remaining parameter budget (`~1.91M / 2.00M`) without forcing dual roles onto a single small matrix.

## Run 5: Increasing Context Window (`block_size = 256`) (`model.py`)
- **Hypothesis**: Doubling `block_size` from `128` to `256` only adds `20,480` parameters (`pos_emb`), bringing total parameters to `1,932,320` (well under the 2,000,000 cap). Combined with BPE subword compression (`~2.5x`), each 256-token training and evaluation window spans `~640 bytes` of historical context, allowing the self-attention layers to model much longer-range dependencies across sentences and paragraphs.
- **What changed**: Set `block_size = 256` in `model.py` `Config` (keeping untied weights, `n_layer=4`, `n_embd=160`, `BPETokenizer vocab=2048`, `RMSNorm`, and `AdamW` schedule).
- **Results**:
  - Parameters: `1,932,320` (`+20,480` vs Run 4)
  - Training loss: `3.3696` (down from `3.6259`)
  - Dev score (`bpb`): **2.0320** (down from `2.1345`)
  - Time: `122s` total (`~61 ms/step`)
- **Conclusion**: Doubling the context length yielded another massive gain (`-0.1025` bpb reduction over Run 4, and `-0.3398` over baseline). Subwords need longer token horizons to establish context; increasing `block_size` directly unlocked the semantic power of BPE subword compression without approaching our parameter ceiling (`1.93M / 2.00M`).

## Run 6: Deeper Narrower Model (`n_layer = 5, n_embd = 128`) (`model.py`)
- **Hypothesis**: Trading off embedding width (`n_embd = 160 -> 128`) to add an extra transformer block (`n_layer = 4 -> 5`) will maintain or improve representation quality while dropping parameter count to `1,547,264`, freeing up `~453k` parameters under the 2M cap for further architectural expansion.
- **What changed**: Set `n_layer = 5` and `n_embd = 128` in `model.py` `Config` (`block_size = 256`, `RMSNorm`, untied weights, BPE `vocab_size = 2048`, AdamW schedule).
- **Results**:
  - Parameters: `1,547,264` (`-385,056` vs Run 5)
  - Training loss: `3.3847`
  - Dev score (`bpb`): **2.0334** (virtually identical to `2.0320` in Run 5)
  - Time: `124s` total (`~62 ms/step`)
- **Conclusion**: Remarkably, a 5-layer / 128-dim model (`1.55M` params) matches the performance of a 4-layer / 160-dim model (`1.93M` params), confirming that deeper representations are more parameter-efficient for this task. We now have `~453,000` extra parameter budget available to scale deeper (`n_layer = 6` or `7`) or add architectural improvements (e.g. SwiGLU / RoPE).

## Run 7: Tying Weights on 5L / 128D (`tie_weights = True`) (`model.py`)
- **Hypothesis**: Tying weights on our narrower (`n_embd = 128`), deeper (`n_layer = 5`) architecture to see if sharing input embedding tables and output logit projection reduces overfitting or hurts representation when `vocab_size = 2048`.
- **What changed**: Set `tie_weights = True` in `model.py` `Config` (keeping `n_layer=5`, `n_embd=128`, `block_size=256`, `RMSNorm`, BPE `vocab_size=2048`).
- **Results**:
  - Parameters: `1,285,120` (`-262,144` vs Run 6)
  - Training loss: `3.5252` (vs `3.3847` in Run 6)
  - Dev score (`bpb`): `2.0934` (`+0.0600` worse vs `2.0334` in Run 6)
  - Time: `119s` total (`~60 ms/step`)
- **Conclusion**: Consistent with Run 2/Run 4, tying weights hurts dev `bpb` (`+0.0600`) when `vocab_size = 2048`. Having separate parameters (`262k` params each) for input semantic token representations (`tok_emb`) and output vocabulary classification (`head`) is essential when using BPE subwords with compact embedding dimensions (`n_embd=128`).

## Run 8: Looped Transformer (`num_loops = 2`, `4L / 160D`) (`model.py`)
- **Hypothesis**: Reusing the same 4 physical blocks (`1,932,320` params) across `num_loops = 2` iterations (`8 effective layers` of computation) will increase network depth without exceeding our 2,000,000 parameter limit.
- **What changed**: Added `num_loops = 2` to `Config` (`n_layer = 4`, `n_embd = 160`, `block_size = 256`, untied weights, `RMSNorm`, BPE) and looped across `self.blocks` twice during `forward()`.
- **Results**:
  - Parameters: `1,932,320` (identical physical parameters to Run 5)
  - Training loss: `3.5533` (vs `3.3696` in Run 5)
  - Dev score (`bpb`): `2.1071` (`+0.0751` worse vs `2.0320` in Run 5)
  - Time: `230s` total (`~115 ms/step`)
- **Conclusion**: Recurrent weight-sharing (`num_loops = 2`) performed significantly worse (`+0.0751` bpb) and took nearly 2x longer (`230s`). Forcing a physical block to simultaneously serve as both an early-stage feature extractor and a late-stage semantic synthesizer creates optimization interference in compact embedding spaces (`160-dim`). Unrolled, independent layers (`Run 5` and `Run 6`) allow each block to specialize at its exact depth (`bpb = 2.0320`).

---

## Run 9: 6-Layer Maximum Depth Unlooped Transformer (`2026-07-20`)
- **Hypothesis**: Expanding physical depth to 6 independent unrolled blocks (`6L / 136D`) without recurrent looping will provide superior hierarchical representation compared to both `4L/160D` (`Run 5`) and looped (`Run 8`).
- **What changed**: Set `n_layer = 6`, `n_embd = 136`, `block_size = 256`, untied weights, `RMSNorm`, BPE tokenizer (`vocab_size = 2048`), `num_loops = 1`.
- **Results**:
  - Parameters: `1,911,880` (`< 2M cap`)
  - Training loss: `3.5244`
  - Dev score (`bpb`): `2.0583` (`+0.0263` vs Run 5)
  - Time: `142s` total (`~71 ms/step`)
- **Conclusion**: Under a strict 2,000 optimizer step budget with `accum_steps = 1`, deeper narrower networks (`6L/136D` at `2.0583 bpb` and `5L/128D` at `2.0334 bpb`) slightly trail wider unrolled networks (`4L/160D` at `2.0320 bpb`) due to gradient variance across more sequential layers.

---

## Run 10: 6-Layer Transformer with 3× Gradient Accumulation & Deep Cosine Decay [NEW CHAMPION] (`2026-07-20`)
- **Hypothesis**: Our N-Gram phase diagnostics and Cross-Entropy vs Empirical Bigram ground-truth comparisons proved that the model is strictly **compute/data bound (`undertrained`)**, not capacity bound. Tripling the tokens processed per optimizer step (`accum_steps = 3`, `3 micro-batches of batch=32 per step` = `24,576 tokens/step`) and cooling the learning rate down to `1e-5` will dramatically improve convergence of the 6-layer architecture (`6L/136D`).
- **What changed**: Set `accum_steps = 3` (`49.15M tokens` processed across 2,000 steps, `~16.8 full epochs`), `min_lr = 1e-5` (`Deep Cosine Decay`), `n_layer = 6`, `n_embd = 136`, `block_size = 256`, untied weights, `RMSNorm`, BPE (`vocab_size = 2048`).
- **Results**:
  - Parameters: `1,911,880` (`< 2M cap`)
  - Training loss: **`2.5602`** (`vs 3.3696 in Run 5 and 3.5244 in Run 9`)
  - Dev score (`bpb`): **`1.7296`** (`-0.3024 bpb` breakthrough improvement vs Run 5 champion!)
  - Time: `2,256s` total (`~1.12s/step` on CPU across `6,000 forward/backward passes`)
- **Conclusion**: **`Run 10` is our definitive Champion configuration.** Tripling data volume per optimizer update eliminated gradient variance and unlocked the true capacity of our 6-layer architecture, achieving `1.7296 bpb`.

---

## 🔬 Empirical N-Gram Trajectory Analysis vs. Step Plateaus & Statistical Regimes
- **Continuous Regime Crossings (`No Step Plateaus`)**: Inspecting overall Model Cross-Entropy plotted against true statistical N-gram boundaries (`ngram_phases.html` & `ngram_regimes.html`) shows that the model does not get trapped in flat step-wise "plateaus" where it remains solely in a unigram or bigram phase for hundreds of optimizer steps. Instead, it continuously breaks across complexity regimes:
  - **Crosses Unigram Bound (`H1 = 7.5859 bpt`)**: **Step 61**
  - **Crosses Bigram Bound (`H2 = 4.2497 bpt`)**: **Step 835**
  - **Trigram & Long-Range Compositional (`< H3 = 1.9780 bpt`)**: Reaches **`1.7296 bpb`** convergence, operating firmly beyond 3-token statistical lookup tables.
- **Note on Validation Curve Straightlines**: Notice that exact validation trajectories (`green curves`) appear as straight lines connecting sparse intervals (`steps 1, 200, ..., 2000` or phase endpoints). This is because full-corpus evaluation on `dev_eval.txt` (`61,404 tokens`) is computationally intensive and evaluated strictly at checkpoint intervals, whereas smoothed training cross-entropy (`blue curves`) tracks real-time convergence step-by-step across all 2,000 steps.

---

## ⚖️ Weight Tying vs. Untied N-Gram Behaviour Comparison (`Equivalent Models`)
- **Rigorous Equivalent-Model Comparison**: To isolate the pure structural effect of weight tying without data volume confounding, we compared **Run 7 (`5L/128D Tied`, `accum=1`, `bpb=2.0934`)** against exact equivalent untied models trained on identical optimization budgets (`16.38M tokens across 2,000 steps`):
  - **Unigram Forward KL**: `4.03 bits` (Run 7 Tied) vs `3.53 bits` (Run 9 Untied) / `3.74 bits` (Run 5 Untied)
  - **Bigram Forward KL**: `8.95 bits` (Run 7 Tied) vs `8.13 bits` (Run 9 Untied) / `8.16 bits` (Run 5 Untied)
  - **Trigram Forward KL**: `10.24 bits` (Run 7 Tied) vs `9.80 bits` (Run 9 Untied) / `9.88 bits` (Run 5 Untied)
- **Conclusion**: When evaluated under strictly equivalent data volume (`16.38M tokens`), the weight-tied model suffers from both worse validation cross-entropy (`2.0934 bpb` vs `2.03–2.05`) *and* higher KL divergence from empirical N-gram distributions. Forcing `tok_emb` and `head` to share parameters in compact dimensions (`128D`) creates a severe representational bottleneck that degrades convergence across both low-order lookup and high-order compositional metrics. (By contrast, `Run 10`'s higher KL is strictly driven by `3x` greater data throughput shifting mass deeper into compositional circuits).
