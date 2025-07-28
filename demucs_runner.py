import argparse
from pathlib import Path
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

input_path = Path(args.input)
output_dir = Path(args.output).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

print(f"🔍 Loading audio from {input_path}")
try:
    waveform, sr = torchaudio.load(input_path)
except Exception as e:
    print(f"❌ Failed to load audio: {e}")
    sys.exit(1)

if sr != 44100:
    print(f"⚠️ Sample rate is {sr}, but Demucs expects 44100 Hz.")
    sys.exit(1)

print("📦 Loading Demucs model...")
model = get_model(name="htdemucs").cpu()

print("🧠 Running model inference...")
try:
    sources = apply_model(model, waveform[None])[0]  # shape: [4, 2, samples]
except Exception as e:
    print(f"❌ Model failed: {e}")
    sys.exit(1)

stem_names = ["drums", "bass", "other", "vocals"]
for i, name in enumerate(stem_names):
    stem_path = output_dir / f"{name}.wav"
    try:
        torchaudio.save(str(stem_path), sources[i], 44100)
        print(f"✅ Saved {name} to {stem_path}")
    except Exception as e:
        print(f"❌ Failed to save {name}: {e}")

print("🎉 All stems written successfully.")
