🧠 Grammar Scoring Engine
This project aims to automatically evaluate the grammar quality of spoken English audio samples using speech recognition, BERT embeddings, language analysis, and regression modeling.

📁 Dataset
Train/Test CSVs: Each row contains an audio filename and (for training data) a human-labeled grammar score.

Audio: .wav files of spoken English from non-native speakers.

🧰 Tools & Libraries
Whisper (ASR model for transcription)

LanguageTool (grammar checker)

Sentence-Transformers (all-MiniLM-L6-v2 for sentence embeddings)

librosa, matplotlib, sklearn, pandas, tqdm

🔍 Feature Extraction
Each audio file is transcribed using Whisper ASR. From the transcript and audio signal, we extract:

Feature	Description
duration	Length of the audio in seconds
num_words	Number of words in the transcript
grammar_errors	Count of grammar mistakes found by LanguageTool
error_rate	Grammar errors divided by total words
words_per_sec	Speaking rate
sentence_count	Number of sentences (based on ., !, ?)
avg_word_length	Average word length
(Optional) bert_embedding	Sentence embedding of transcript using BERT (if added)
⚙️ Model Training
A Random Forest Regressor is trained on the extracted features to predict grammar scores:

train_test_split: 80/20 for training/validation

Evaluation metrics:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

📈 Sample Results
text
Copy
Edit
Validation RMSE: 0.423
Validation MAE : 0.318
Plot of actual vs predicted validation scores:

<p align="center"> <img src="assets/prediction_scatter.png" alt="Prediction vs Actual Plot" width="500"/> </p>
🧪 Test Inference
The test set audio is processed using the same pipeline. Final predictions are saved to:

bash
Copy
Edit
submission.csv
Format:

filename	label
audio1.wav	3.7
audio2.wav	2.5
...	...
💡 Future Improvements
Add BERT sentence embeddings for richer semantic features

Extract fluency or pause features using prosody

Fine-tune models or use gradient boosting (e.g., XGBoost)


