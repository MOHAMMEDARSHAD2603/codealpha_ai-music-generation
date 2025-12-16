# 🎶 AI Music Generation with LSTM | CodeAlpha Internship

This project trains a deep learning model to generate music using classical MIDI files. It was built as part of the CodeAlpha internship and demonstrates the full pipeline from data collection to music playback.

---

## 📁 Project Structure
- `music_gen.py` — trains the LSTM model and generates new music.
- `midi_songs/` — folder containing training MIDI files (classical pieces).
- `generated_music.mid` — output MIDI file from the model.
- `play_midi.py` — script to auto-play the generated MIDI.
- `inspect_midi.py` — prints note sequences for debugging.
- `midi_to_wav.py` — optional script to convert MIDI to WAV using FluidSynth.
- `app.py` — Streamlit app for interactive demo.
- `requirements.txt` — dependencies for running the project.

---

## 🧠 Model Overview
- Framework: PyTorch  
- Architecture: LSTM (Recurrent Neural Network)  
- Input: Preprocessed note sequences from MIDI files  
- Output: New note sequences saved as MIDI  

---

## 🚀 Getting Started

**Prerequisites**
- Python 3.12
- pip (Python package manager)

**Installation**
```bash
git clone https://github.com/MOHAMMEDARSHAD2603/codealpha_ai-music-generation.git
cd codealpha_ai-music-generation
pip install -r requirements.txt

Run Training + Generation
python music_gen.py


Play Output
python play_midi.py


Optional: Convert to WAV
python midi_to_wav.py


🎧 Usage
- Collect MIDI files in midi_songs/
- Train the LSTM model with music_gen.py
- Generate new music → saved as generated_music.mid
- Play or convert to WAV for polished audio
- Try the interactive demo via app.py (Streamlit)

🔮 Future Improvements
- Add GAN-based music generation for richer compositions
- Build a Streamlit web app with real-time playback
- Expand dataset with multiple genres (jazz, pop, Indian classical)
- Add download button for generated audio in the app
- Deploy as a mobile app with PyTorch Mobile


📌 Author
Mohammed Arshad.R
Coimbatore, Tamil Nadu, India
CodeAlpha Internship Project, 2025



