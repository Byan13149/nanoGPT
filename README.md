# NanoGPT Variants

This version adds several modules to `model.py`, including **RoPE** (and its variants), **N-gram adapter**, and the **Memory-enhanced block**, along with the corresponding parameter definitions in `train.py`.

## Usage

1. **Prepare the dataset**

   Download `enwik8.zip` into `data/enwik8/`, then run:

   ```bash
   python data/enwik8/prep_enwik8.py

2. **Train different model variants**

   ***Standard NanoGPT***
   ```bash
   python train.py config/train_enwik8_std.py
   ```

   ***Memory-enhanced NanoGPT***
   ```bash
   python train.py config/train_enwik8_mem.py
   ```

   ***NanoGPT with RoPE***
   ```bash
   python train.py config/train_enwik8_rope.py
   ```