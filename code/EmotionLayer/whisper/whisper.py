from groq import Groq #type: ignore 
from pathlib import Path
import time 

from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def get_transcribe_audio(audio_path):
    transcription = client.audio.transcriptions.create(
        file= audio_path,
        model="whisper-large-v3-turbo",
        temperature=0,
        language="it",
    )
    return transcription.text

def main():
    audio_path = Path(f'C:/Users/Roberto/OneDrive/Desktop/AffectiveComputing/audio/dataset/AI4SER_dataset/data/07/ang_04_07.wav')
    start_time = time.time()
    result = get_transcribe_audio(audio_path)
    end_time = time.time()
    duration_sec = end_time - start_time

    print(f"Trascrizione completa in {duration_sec:.2f} secondi:/n{result}")

if __name__ == "__main__":
    main()
