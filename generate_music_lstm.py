"""
Music Generation with AI (LSTM version)
-----------------------------------------
Task 3: Music Generation with AI

Full deep-learning pipeline:
  1. Collect MIDI music data (a built-in seed corpus of note events, or your
     own .mid files with --midi-dir).
  2. Preprocess the data into note/duration sequences using `music21`.
  3. Build a deep learning model (an LSTM, via Keras/TensorFlow) that learns
     music patterns from fixed-length windows of notes.
  4. Train the model on the dataset to predict the next note in a sequence.
  5. Generate a brand-new sequence by sampling from the trained model, then
     convert it back to MIDI so you can play it.

Usage:
    python generate_music.py                          # built-in seed corpus
    python generate_music.py --midi-dir ./my_midis     # train on your own MIDI files
    python generate_music.py --epochs 60 --length 80 --out song.mid
"""

import argparse
import random
from pathlib import Path

import numpy as np
from music21 import stream, note, chord, duration as m21duration, midi, converter, tempo as m21tempo

# Keras / TensorFlow -- the actual deep learning model
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# 1. Data collection: a built-in seed corpus (several short melodic phrases,
#    including repeats/transpositions so there's enough data for an LSTM to
#    find real patterns in). Each element is (pitch_name, quarterLength).
#    Point --midi-dir at a folder of real .mid files for a proper dataset.
# ---------------------------------------------------------------------------
def _phrase_c_major():
    return [
        ("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0),
        ("B4", 0.5), ("G4", 0.5), ("E4", 1.0), ("D4", 1.0),
        ("C4", 1.0), ("D4", 1.0), ("E4", 1.0), ("F4", 1.0),
        ("G4", 2.0), ("E4", 1.0), ("C4", 1.0),
    ]


def _phrase_a_minor():
    return [
        ("A4", 1.0), ("C5", 1.0), ("E5", 1.0), ("A4", 1.0),
        ("G4", 0.5), ("E4", 0.5), ("C4", 1.0), ("D4", 1.0),
        ("E4", 1.0), ("F4", 1.0), ("G4", 1.0), ("A4", 2.0),
        ("E4", 1.0), ("C4", 1.0),
    ]


def _phrase_g_major():
    return [
        ("G4", 1.0), ("B4", 1.0), ("D5", 1.0), ("G5", 1.0),
        ("F5", 0.5), ("D5", 0.5), ("B4", 1.0), ("A4", 1.0),
        ("G4", 1.0), ("A4", 1.0), ("B4", 1.0), ("C5", 1.0),
        ("D5", 2.0), ("B4", 1.0), ("G4", 1.0),
    ]


def build_seed_corpus():
    """Combine several phrases (with a couple of repeats) into one training corpus."""
    events = []
    for phrase in [_phrase_c_major(), _phrase_a_minor(), _phrase_g_major(),
                   _phrase_c_major(), _phrase_g_major(), _phrase_a_minor()]:
        events.extend(phrase)
    return events


def load_corpus_from_midi_dir(midi_dir: str):
    """Preprocess real MIDI files into (pitch, duration) sequences using music21."""
    events = []
    midi_paths = list(Path(midi_dir).glob("*.mid")) + list(Path(midi_dir).glob("*.midi"))
    if not midi_paths:
        raise FileNotFoundError(f"No .mid/.midi files found in {midi_dir}")

    for path in midi_paths:
        score = converter.parse(str(path))
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                events.append((element.pitch.nameWithOctave, float(element.duration.quarterLength)))
            elif isinstance(element, chord.Chord):
                events.append((element.root().nameWithOctave, float(element.duration.quarterLength)))
    return events


# ---------------------------------------------------------------------------
# 2. Preprocessing: turn (pitch, duration) events into integer-encoded
#    fixed-length windows for supervised next-note prediction.
# ---------------------------------------------------------------------------
def build_vocab(events):
    vocab = sorted(set(events))
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for i, n in enumerate(vocab)}
    return vocab, note_to_int, int_to_note


def make_training_windows(events, note_to_int, seq_len: int):
    encoded = [note_to_int[e] for e in events]
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i:i + seq_len])
        y.append(encoded[i + seq_len])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# 3 & 4. Model: an Embedding + LSTM network that learns to predict the next
#    note given the previous `seq_len` notes, trained on the dataset.
# ---------------------------------------------------------------------------
def build_lstm_model(vocab_size: int, seq_len: int):
    model = keras.Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=32),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(128, activation="relu"),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# 5. Generation + MIDI export
# ---------------------------------------------------------------------------
def sample_with_temperature(probs, temperature: float = 1.0):
    probs = np.asarray(probs).astype("float64")
    probs = np.log(probs + 1e-9) / temperature
    probs = np.exp(probs)
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)


def generate_sequence(model, seed_window, int_to_note, length: int, temperature: float = 1.0):
    window = list(seed_window)
    generated = list(seed_window)

    while len(generated) < length:
        x = np.array(window[-len(seed_window):]).reshape(1, -1)
        probs = model.predict(x, verbose=0)[0]
        next_idx = sample_with_temperature(probs, temperature)
        generated.append(next_idx)
        window.append(next_idx)

    return [int_to_note[i] for i in generated[:length]]


def sequence_to_midi(events, out_path: str, tempo_bpm: int = 100):
    part = stream.Part()
    part.append(m21tempo.MetronomeMark(number=tempo_bpm))

    for pitch_name, dur in events:
        n = note.Note(pitch_name)
        n.duration = m21duration.Duration(dur)
        part.append(n)

    score = stream.Score()
    score.append(part)
    mf = midi.translate.streamToMidiFile(score)
    mf.open(out_path, "wb")
    mf.write()
    mf.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a melody with a trained LSTM and export it to MIDI.")
    parser.add_argument("--midi-dir", default=None, help="Folder of .mid/.midi files to train on (optional)")
    parser.add_argument("--seq-len", type=int, default=8, help="Number of previous notes the LSTM sees as context")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--length", type=int, default=60, help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling randomness (higher = more random)")
    parser.add_argument("--tempo", type=int, default=100, help="Tempo in BPM")
    parser.add_argument("--out", default="generated_music.mid", help="Output MIDI file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("Step 1-2: Collecting & preprocessing training data...")
    if args.midi_dir:
        events = load_corpus_from_midi_dir(args.midi_dir)
        print(f"  Loaded {len(events)} note events from {args.midi_dir}")
    else:
        events = build_seed_corpus()
        print(f"  Using built-in seed corpus ({len(events)} note events)")

    vocab, note_to_int, int_to_note = build_vocab(events)
    print(f"  Vocabulary size: {len(vocab)} unique (pitch, duration) pairs")

    seq_len = min(args.seq_len, max(2, len(events) // 3))
    X, y = make_training_windows(events, note_to_int, seq_len)
    print(f"  Built {len(X)} training windows of length {seq_len}")

    print("Step 3: Building the LSTM model...")
    model = build_lstm_model(vocab_size=len(vocab), seq_len=seq_len)
    model.summary()

    print(f"Step 4: Training on note patterns for {args.epochs} epochs...")
    model.fit(X, y, epochs=args.epochs, batch_size=16, verbose=2)

    print(f"Step 5a: Generating a new {args.length}-note sequence from the trained model...")
    start_idx = random.randint(0, len(X) - 1)
    seed_window = X[start_idx]
    generated = generate_sequence(model, seed_window, int_to_note, args.length, args.temperature)

    print(f"Step 5b: Writing MIDI file to {args.out} ...")
    sequence_to_midi(generated, args.out, tempo_bpm=args.tempo)

    print("\nDone! Generated sequence (pitch, duration in quarter notes):")
    print(generated)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
