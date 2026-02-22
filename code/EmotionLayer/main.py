import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import custom functions
from prosodyAnalysis.run_prosody import get_prosody_info
from voiceAnalysis.run_emotion import get_emotion_info
from whisper.whisper import get_transcribe_audio


def numpy_converter(obj):
    """Converts NumPy objects into standard Python types for JSON serialization."""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def measure_execution(func, *args):
    "Executes a function and returns (result, elapsed_time)."
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start


def get_audio_analysis_parallel(audio_path):
    with ThreadPoolExecutor() as executor:
        # Task 1: Voice Sentiment Analysis (wrapped for timing)
        future_emotion = executor.submit(measure_execution, get_emotion_info, audio_path)
        
        # Task 2: Arousal/Valence (wrapped for timing)
        future_prosody = executor.submit(measure_execution, get_prosody_info, audio_path)
        
        # Task 3: Text transcription
        future_text = executor.submit(measure_execution, get_transcribe_audio, audio_path)

        # Collect results and timings
        # measure_execution returns a tuple: (data, time)
        voice_sentiment_list, time_emotion = future_emotion.result()
        (arousal, dominance, valence), time_prosody = future_prosody.result()        
        transcription, time_trascription = future_text.result()

    # Create a dedicated dictionary for timings for cleanliness
    timings = {
        "voice_sentiment_latency": time_emotion,
        "prosody_latency": time_prosody,
        "transcription_latency": time_trascription,
    }

    result = {
        "vocal_sentiment": voice_sentiment_list[:2],
        "prosody": {
            "arousal": arousal,
            "dominance": dominance,
            "valence": valence
        },
        "transcription": transcription,
        "timings": timings
    }
        
    return result

def single_inference():
    audio_path = Path("<your_audio_path>")
    
    print("Starting parallel analysis...")
    global_start = time.time()
    
    result = get_audio_analysis_parallel(audio_path)
    
    global_end = time.time()
    
    # Add the global total time
    result["timings"]["total_parallel_execution"] = global_end - global_start

    # Use the numpy converter to avoid float32 errors
    print(json.dumps(result, indent=4, default=numpy_converter))


def audio_folder_inference():
    base_path = Path('your_folder_path')

    print(f"Starting audio analysis in '{base_path}'...")
    
    for audio_file in base_path.iterdir():
        if audio_file.is_file() and audio_file.suffix.lower() == '.wav' or audio_file.suffix.lower() == '.mp3':
            
            print(f"\n--- Processing: {audio_file.name} ---")
            
            start = time.time()    
            try:
                result = get_audio_analysis_parallel(audio_file)
                
                end = time.time()
                result["timings"]["total_parallel_execution"] = end - start

                # Print the JSON
                print(json.dumps(result, indent=4, default=numpy_converter))
                
            except Exception as e:
                print(f"Error during analysis of {audio_file.name}: {e}")

if __name__ == "__main__":
    single_inference()