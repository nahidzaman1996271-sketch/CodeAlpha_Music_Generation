# Music Generation with AI

Two versions are included. **Use `generate_music_lstm.py`** — it's the one
that matches the checklist ("build a deep learning model using RNNs like
LSTM").

## Files
- `generate_music_lstm.py` — ✅ **the checklist-compliant version.** Full
  pipeline: collect → preprocess → build & train a real Keras/TensorFlow
  LSTM → generate → export to MIDI.
- `generated_music_lstm.mid` — example output from the LSTM version.
- `generate_music.py` — an earlier, lighter Markov-chain version (kept for
  reference/comparison). Trains instantly but is **not** a deep learning
  model, so it doesn't satisfy the "build an RNN/LSTM or GAN" requirement.
- `generated_music.mid` — example output from the Markov version.

## How `generate_music_lstm.py` works (matches the task checklist)
1. **Collect MIDI data** — a built-in seed corpus of three short melodic
   phrases (repeated for more training data), or point `--midi-dir` at your
   own folder of `.mid` files (classical, jazz, whatever you like).
2. **Preprocess** — `music21` parses/represents notes as `(pitch, duration)`
   pairs, which are integer-encoded into fixed-length training windows.
3. **Build a deep learning model** — a Keras `Sequential` model:
   `Embedding → LSTM(128) → LSTM(128) → Dense(128) → Dense(softmax)`.
4. **Train the model** — trained with `model.fit()` for `--epochs` passes
   over the dataset to predict the next note given the previous ones
   (this is real gradient-descent training, not a lookup table).
5. **Generate & convert to MIDI** — samples a new sequence note-by-note from
   the trained model's predicted probabilities, then `music21` writes it
   out as a `.mid` file.

## Run it
```bash
pip install music21 tensorflow
python generate_music_lstm.py
python generate_music_lstm.py --midi-dir ./my_midis     # train on your own MIDI files
python generate_music_lstm.py --epochs 100 --length 80 --temperature 1.1 --out song.mid
```

Flags:
- `--epochs` — training passes (default 60; more = better-fitted patterns, up to a point)
- `--seq-len` — how many previous notes the LSTM looks at for context (default 8)
- `--temperature` — sampling randomness (lower = safer/more repetitive, higher = more chaotic)
- `--length` — how many notes to generate
- `--seed` — fix for reproducible output

## Honest limitation
The built-in corpus is small (~90 notes), so the LSTM has limited data to
learn from — training loss/accuracy improves clearly over epochs (you'll
see this printed live), but a real project would train on a proper MIDI
dataset (hundreds of songs) via `--midi-dir` for noticeably more musical
results.
