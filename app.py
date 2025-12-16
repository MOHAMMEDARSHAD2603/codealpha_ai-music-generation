import streamlit as st
from music21 import converter

st.title("🎶 AI Music Generator")

if st.button("Generate Music"):
    # Run your model here
    st.success("Music generated!")
    st.audio("generated_music.mid")