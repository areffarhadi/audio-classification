# finetuning Resnet293 model for audio classification
Using this script, we can implement several scenarios in audio classification, such as speaker identification, language recognition, emotion recognition, sentiment analysis, and more, using [ResNet293](https://drive.google.com/file/d/1nZFMhtAXlH5fiK5MrFW5HfVsraqRmhe2/view?usp=sharing) originally trained for speaker recognition in the VoxBlink2 project.
 

#To fine-tune and test ResNet293 model:
1. download the pretrained model from the link we presented above.
2. make manifest using `[wav_manifest.py](https://github.com/areffarhadi/audio-classification/blob/main/ResNet293-model/manifest_voice.py)'
The manifest for feeding wav data must be like [train_voice_emotion.csv](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/train_voice_emotion.csv) file.



