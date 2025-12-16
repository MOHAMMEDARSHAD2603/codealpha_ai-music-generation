import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the MIDI file
pygame.mixer.music.load("generated_music.mid")

# Play and record to WAV
pygame.mixer.music.play()

print("Playing generated_music.mid...")

# Keep the script alive until playback finishes
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)