import streamlit as st
import whisper
import numpy as np
from st_audiorec import st_audiorec
import tempfile
import librosa
from summarizing import Summarizer
import os

st.title('Whisper for Speech to text')

# Option to upload or record audio
audio_option = st.radio("Select Audio Source", ("Upload Audio", "Record Audio"))

# Placeholder for audio file or recording widget
audio_data = None

if audio_option == "Upload Audio":
    # Upload audio file
    audio_file = st.file_uploader('Upload Audio', type=['wav', 'mp3', 'ogg'])
    
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav", start_time=0)
        audio_data, _ = librosa.load(audio_file, sr=16000)

elif audio_option == "Record Audio":
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        # Convert recorded audio bytes to NumPy array
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(wav_audio_data)
            temp_file_path = temp_file.name

        # Convert recorded audio bytes to NumPy array
        audio_data, _ = librosa.load(temp_file_path, sr=16000)

# Load model
@st.cache_data
def load_whisper():
    model = whisper.load_model('base')
    return model

# Check if the model is loaded
if 'model' not in st.session_state:
    st.session_state.model = load_whisper()

# Button to load the Whisper model
if st.sidebar.button('Load Whisper model'):
    model = load_whisper()
    st.sidebar.success("Whisper Model Loaded")

text =''
# Button to transcribe audio
if st.sidebar.button("Transcribe Audio"):
    if audio_data is not None:
        try:
            st.sidebar.success('Transcribing Audio')
            transcription = st.session_state.model.transcribe(audio_data.astype(np.float32))
            st.sidebar.success('Transcription complete')
            st.markdown(transcription['text'])
            text = transcription['text']
        except Exception as e:
            st.error(f'Error transcribing audio:', e)
        # finally:
        #     # Delete the temporary file after processing
        #     if temp_file_path and os.path.exists(temp_file_path):
        #         os.remove(temp_file_path)
        #         # st.write(f"Deleted temporary file: {temp_file_path}")
    else:
        st.sidebar.error("Please upload or record an audio file")

#summarizing audio using facebook/bart-large-cnn
        
def summarize(text):
    if text:        
        summarize_class = Summarizer(text)    
        summary =summarize_class.summary()
        return summary
    else:
        return None

st.markdown("Summary")

if summarize(text) == None:
    st.markdown("")

else:
    st.markdown(summarize(text=text))

#below script is good for cleanup 
    
# temp_dir = tempfile.gettempdir()


# files_to_delete = [f for f in os.listdir(temp_dir) if f.endswith(".wav")]

# for file_to_delete in files_to_delete:
#     file_path = os.path.join(temp_dir, file_to_delete)
#     try:
#         os.remove(file_path)
#         print(f"Deleted file: {file_path}")
#     except Exception as e:
#         print(f"Error deleting file {file_path}: {e}")