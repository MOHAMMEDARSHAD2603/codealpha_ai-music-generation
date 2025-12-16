from music21 import converter

# Load the generated MIDI
s = converter.parse('generated_music.mid')

# Show the notes and chords in text format
s.show('text')

# Play the MIDI using your system's default player
s.show('midi')