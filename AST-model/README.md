## finetuning AST model for audio classification
Using this script, we can implement several scenarios in audio classification, such as speaker identification, language recognition, emotion recognition, sentiment analysis and more, using [AST](https://github.com/YuanGongND/ast).

We fine-tuned this model for `speech/non-speech` segment recognition. The fine-tuned model is available [here](https://drive.google.com/file/d/1tIoexZMWJzBr3LbTK8YX1G8RDZCVwbRt/view?usp=sharing). To use this model, follow the test section of the description below.

# To fine-tune and test AST model:
1. download the pretrained model [here](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)
2. making manifests using [manifest_maker.py](https://github.com/areffarhadi/audio-classification/blob/main/AST-model/manifest_maker.py) for wav files and [manifest_maker_mp4.py](https://github.com/areffarhadi/audio-classification/blob/main/AST-model/manifest_maker_mp4.py) for mp4 files.
3. fine-tune the model using [AST_fine_tuning_AudioSet.py](https://github.com/areffarhadi/audio-classification/blob/main/AST-model/AST_fine_tuning_AudioSet.py) by editing the script to set the manifests and pre-trained model addresses before running!
4. inference the model using [ast_inference_with_manifest.py](https://github.com/areffarhadi/audio-classification/blob/main/AST-model/ast_inference_with_manifest.py) and see the result in the output CSV file.
5. remove unwanted files with specific labels recognized by the modelby [remove_but_speech.py](https://github.com/areffarhadi/audio-classification/blob/main/AST-model/remove_but_speech.py) (for example, "music" and "non-speech" files in the fine-tuned model)
6. In addition, remove or filter out files that are recognized in the goal class but with a score less than the threshold by [filter_csv_out_0.5.py](https://github.com/areffarhadi/audio-classification/blob/main/AST-model/filter_csv_out_0.5.py) (for example, for the "speech" class in our fine-tuned model, if the score is less than 0.5). We do not need them. 




