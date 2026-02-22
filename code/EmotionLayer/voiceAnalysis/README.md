# Wav2Vec2.0 Italian Emotion Prediction

This project contains the inference code for a model based on the **Wav2Vec2** architecture, fine-tuned for emotion recognition in Italian speech. The model analyzes audio files and returns predictive values for all emotions:

* anger
* disgust
* fear
* joy
* neutrality
* sadness
* surprise

## 📁 Project Structure

* `prediction.py`: Main script to run inference on an audio file. It loads the model, extracts features via `Wav2Vec2FeatureExtractor`, and returns the predicted values.
* `src/models.py`: Contains the definition of the custom model architecture (`Wav2Vec2ForSpeechClassification`).
* `src/modeling_output.py`: Defines the dataclass to structure the model's output.
* `model/`: Target folder where the trained model must be saved (not included by default).

## 🚀 Usage Guide

### 1. Environment Setup

First, create a virtual environment and install the necessary dependencies using the `requirements.txt` file:

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

* **Download (Recommended):** The pre-trained Italian emotion recognition model is available on Hugging Face. Download the files from [abernardini-dev/wav2vec2-large-xlsr-53-italian-emotion-recognition](https://huggingface.co/abernardini-dev/wav2vec2-large-xlsr-53-italian-emotion-recognition).
* **Train from Scratch:** Alternatively, train the model using the dedicated repository: [Wav2Vec2.0-Italian-emotion](https://github.com/abernardini-unimi/Wav2Vec2.0-Italian-emotion).

Once you have the model files, **place them inside the `model/` folder** of this project.

### 3. Running the Prediction

You can test an audio file by modifying the script.

Open `prediction.py` and modify the `file_audio` variable in the `if __name__ == "__main__":` block with your desired default path. Then run:

```bash
python prediction.py

```

The output will be a properly formatted JSON object containing the metrics:

```json
[
    {
        "Emotion": "fear",
        "Score": 0.6297
    },
    {
        "Emotion": "neutrality",
        "Score": 0.2024
    },
    {
        "Emotion": "sadness",
        "Score": 0.0733
    },
    {
        "Emotion": "surprise",
        "Score": 0.0358
    },
    {
        "Emotion": "anger",
        "Score": 0.0264
    },
    {
        "Emotion": "joy",
        "Score": 0.0173
    },
    {
        "Emotion": "disgust",
        "Score": 0.0150
    }
]

```