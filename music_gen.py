import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# STEP 1: Load MIDI Data
# -----------------------------
notes = []
files = glob.glob("midi_songs/*.mid")
if len(files) == 0:
    raise FileNotFoundError("No MIDI files found in 'midi_songs'. Add .mid files and rerun.")

for file in files:
    try:
        midi = converter.parse(file)
    except Exception as e:
        print(f"Skipping '{file}' due to parse error: {e}")
        continue

    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

if len(notes) < 60:
    raise ValueError("Not enough notes extracted. Add more MIDI files or use simpler MIDI tracks.")

# -----------------------------
# STEP 2: Preprocess Notes
# -----------------------------
unique_notes = sorted(set(notes))
note_to_int = {n: i for i, n in enumerate(unique_notes)}
int_to_note = {i: n for n, i in note_to_int.items()}

sequence_length = 50
input_sequences = []
output_notes = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    input_sequences.append([note_to_int[n] for n in seq_in])
    output_notes.append(note_to_int[seq_out])

X = torch.tensor(input_sequences, dtype=torch.long)
y = torch.tensor(output_notes, dtype=torch.long)

# -----------------------------
# STEP 3: Define LSTM Model
# -----------------------------
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

model = MusicLSTM(len(unique_notes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# STEP 4: Train Model (simplified)
# -----------------------------
epochs = 10  # start small; increase later
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# -----------------------------
# STEP 5: Generate Music
# -----------------------------
start = np.random.randint(0, len(input_sequences)-1)
pattern = input_sequences[start].copy()
prediction_output = []

model.eval()
with torch.no_grad():
    for _ in range(100):  # generate 100 notes
        input_seq = torch.tensor([pattern], dtype=torch.long)
        prediction = model(input_seq)
        index = torch.argmax(prediction).item()
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]

# -----------------------------
# STEP 6: Convert to MIDI
# -----------------------------
output_notes_stream = []
for pattern in prediction_output:
    if '.' in pattern or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_in_chord = [int(n) for n in notes_in_chord]
        chord_notes = [note.Note(n) for n in notes_in_chord]
        new_chord = chord.Chord(chord_notes)
        new_chord.quarterLength = 0.5
        output_notes_stream.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.quarterLength = 0.5
        output_notes_stream.append(new_note)

midi_stream = stream.Stream(output_notes_stream)
midi_stream.write('midi', fp='generated_music.mid')
print("Generated music saved as generated_music.mid")

