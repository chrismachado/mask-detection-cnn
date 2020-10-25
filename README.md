# Face Mask Recognition

## Usage

Cloning this repo using
```commandline
git clone https://github.com/chrismachado/mask-detection-cnn
```

Install python dependencies with pip
```commandline
pip install -r requirements.txt 
```
Then, if you just want to train and test the model, just type
```commandline
python train_mask_detector.py -d dataset 
```
Or, you can run pre-trained model in output folder by using the following command
```commandline
python detect_mask_image.py -i examples/example_01.png
```

For more information about the script, you can use help command
```commandline
python detect_mask_image.py -h
```

## Author
- Christiano Machado (christianomachado10@gmail.com | github.com/chrismachado)

## References
- [Prajna’s GitHub repository](https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator)
- [Adrian Rosebrock's Article](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)