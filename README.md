# Wav2vec2 and Whisper models for audio classification

In this repo, we prepare the code to fine-tune four models for audio classification, which could be used in different scenarios, such as speaker identification, language recognition, emotion recognition, sentiment analysis, speaking-style recognition, speech/non-speech classification, and more.

These four models have good performance in their fields. 

Whisper: originally for ASR

ResNet293: originally for speaker recognition

Wav2Vec2: good performance in audio classification

AST: originally for audio event classification


  # to do list for Whisper and Wav2vec2 models:

1. The manifest for feeding wav data must be like [train_voice_emotion.csv](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/train_voice_emotion.csv) file.


2. For fine-tuning the [Whisper](https://github.com/openai/whisper) model for audio classification: [Whisper_Emotion.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/Whisper_Emotion.py) <be>

3. For fine-tuning the [Wav2Vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) for audio classification: [wav2vec_Emotion.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_Emotion.py)


in addition we have [wav2vec_Emotion_specaugm.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_Emotion_specaugm.py) for utilizing SpecAugment as augmentation technique, [wav2vec_embeding.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_embeding.py) for extract and save feature embedings and [wav2vec_emb_score.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_emb_score.py) for extracting scores for each wav file.

Please use [slurm_run.sh](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/slurm_run.sh) to run the scripts with Slurm.



