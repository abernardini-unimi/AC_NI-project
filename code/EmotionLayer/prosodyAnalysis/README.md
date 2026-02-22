# Wav2Vec2.0 Italian Prosody Value Prediction

This project contains the inference code for a model based on the **Wav2Vec2** architecture, optimized for the continuous classification (regression) of emotional and prosodic traits in speech. The model analyzes audio files and returns predictive values for the three classic dimensions of emotions:

* **Arousal** * **Valence** * **Dominance** ## 📁 Project Structure
* `prediction.py`: Main script to run inference on an audio file. It loads the model, extracts features via `Wav2Vec2FeatureExtractor`, and returns the predicted values.
* `src/models.py`: Contains the definition of the custom model architecture (`Wav2Vec2ForSpeechClassification`), the classification head, and the **CCC** (Concordance Correlation Coefficient) loss function.
* `src/modeling_output.py`: Defines the dataclass to structure the model's output.
* `model/`: Target folder where the trained model must be saved (not included by default).

## 🚀 Usage Guide

### 1. Environment Setup

First, create a virtual environment and install the necessary dependencies using the `requirements.txt` file (which should include `torch`, `librosa`, and `transformers`).

```bash
# Create the virtual environment
python -m venv venv

# Activation (Linux/macOS)
source venv/bin/activate
# Activation (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Model Download and Placement

This repository strictly handles inference. Before running the prediction script, you need the trained model weights. You can either download the pre-trained model or train one yourself:

* **Download (Recommended):** The pre-trained Italian prosody model is available on Hugging Face. Download the files from [abernardini-dev/wav2vec2-large-xlsr-53-italian-pad](https://huggingface.co/abernardini-dev/wav2vec2-large-xlsr-53-italian-pad).
* **Train from Scratch:** Alternatively, train the model using the dedicated repository: [Wav2Vec2.0-Italian-prosody](https://github.com/abernardini-unimi/Wav2Vec2.0-Italian-prosody).

Once you have the files, **place them inside the `model/` folder** of this project.

### 3. Running the Prediction

You can test an audio file in two ways.

**Option A: Passing the file path via command line**

```bash
python prediction.py "path/to/your/audio_file.wav"

```

**Option B: Modifying the script**
Open `prediction.py` and modify the `file_audio` variable in the `if __name__ == "__main__":` block with your desired default path. Then run:

```bash
python prediction.py

```

The output will be a properly formatted JSON object containing the three metrics:

```json
{
    "arousal": 0.1234,
    "valence": 0.5678,
    "dominance": 0.9012
}

```