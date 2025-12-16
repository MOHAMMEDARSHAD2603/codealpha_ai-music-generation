from music21 import converter

# Load the generated MIDI
s = converter.parse('generated_music.mid')

# Play it using your system's default MIDI player
s.show('midi')