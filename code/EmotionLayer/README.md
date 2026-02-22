# 🗣️ EmotionLayer

EmotionLayer is an integrated system that analyzes speech to detect prosodic values and emotions from audio files. The pipeline runs parallel inference to extract vocal information, transcribe speech, and process text sentiment, returning a structured and comprehensive output along with execution latencies.

The system relies on two dedicated submodules, **ProsodyAnalysis** and **VoiceAnalysis**, which use models based on the `Wav2Vec2.0` architecture specifically trained for the Italian language (more information can be found in the READMEs of their respective folders).

## ⚡ Key Features

* **Prosody Analysis**: Extraction of continuous values for *Valence*, *Arousal*, and *Dominance*.
* **Vocal Emotion Analysis**: Prediction of the main emotion expressed in the audio.
* **Speech-to-Text**: Automatic audio transcription using Whisper.
* **Text Sentiment**: Sentiment analysis applied to the text transcription.
* **Parallel Execution**: Utilization of `ThreadPoolExecutor` to minimize processing times.

## ⚙️ Installation

To run EmotionLayer, you need to isolate dependencies by creating a virtual environment and installing the required packages.

1. **Create a virtual environment**:

```bash
python -m venv venv

```

2. **Activate the virtual environment**:

* On Windows: `venv\Scripts\activate`
* On macOS/Linux: `source venv/bin/activate`

3. **Install the requirements**:

```bash
pip install -r requirements.txt

```

4. **Configure environment variables**:
Rename the `env-example` file to `.env` and insert your Groq API key inside it to enable transcriptions and text analysis features:

```bash
mv env-example .env

```

*(Open the `.env` file and add your key: `GROQ_API_KEY=your_key_here`)*

## 🚀 Usage

### Running the Full Pipeline (EmotionLayer)

The main entry point for execution is the `main.py` file. The script is configured to process entire folders of audio files or single tracks.

Before running it, open `main.py` and make sure to update the paths by inserting the path of the folder or audio file you want to process:

* In the `single_inference()` function to test a single file.
* In the `audio_folder_inference()` function to process a directory of `.wav` or `.mp3` files.

To start the full analysis, run:

```bash
python main.py

```

### Testing Individual Modules

If you want to test the individual components independently, you can run them as separate Python modules from the root directory:

* **Test Prosody Extraction**:
```bash
python -m prosodyAnalysis.run_prosody

```


* **Test Emotion Extraction**:
```bash
python -m voiceAnalysis.run_emotion

```


* **Test Audio Transcription**:
```bash
python -m whisper.whisper

```



## 📊 Output Example

The script returns a well-formatted JSON object in the console for each processed file, containing the results of all analyses and performance metrics.

```json
{
    "vocal_sentiment": [
        {
            "Emotion": "anger",
            "Score": "75%"
        },
        {
            "Emotion": "sad",
            "Score": "10%"
        }
    ],
    "prosody": {
        "arousal": 0.95,
        "dominance": 0.90,
        "valence": 0.05
    },
    "transcription": "..."
}

```