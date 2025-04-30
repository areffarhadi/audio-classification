# finetuning Resnet293 model for audio classification
Using this script, we can implement several scenarios in audio classification, such as speaker identification, language recognition, emotion recognition, sentiment analysis, and more, using [ResNet293](https://drive.google.com/file/d/1nZFMhtAXlH5fiK5MrFW5HfVsraqRmhe2/view?usp=sharing) originally trained for speaker recognition in the VoxBlink2 project.
 

#To fine-tune and test ResNet293 model:
1. download the pretrained model from the link we presented above.
2. make manifest using [wav_manifest.py](https://github.com/areffarhadi/audio-classification/blob/main/ResNet293-model/manifest_voice.py) and repeat it for all train, valid and test sets.
3. in the `conf/resnet293_4.yaml` update the pathes to the manifests and directory of pretrained model
4. fine tune the model by running [fine_tune_frozen_resnet293.py](https://github.com/areffarhadi/audio-classification/blob/main/ResNet293-model/fine_tuning_frozen_Resnet293.py)
5. finally test the model using []()



