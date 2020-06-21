# MODEL IS BASED ON 
https://github.com/ajaichemmanam/simple_bodypix_python

# Getting Started
1. Install requirements in virtual environment

```
python -m venv venvSBP
. venvSBP/bin/activate
pip install -r requirements.txt
```

2. Download tfjs models for bodypix.
```bash
# For example, download a ResNet50-based model to ./bodypix_resnet50_float_model-stride16
$ ./get-model.sh bodypix/resnet50/float/model-stride16
```

3. Set path to models and image for inference in .py files
```py
videoPath = './awesome_img.jpg'
modelPath = './bodypix_resnet50_float_model-stride16/'
```
4. Run the script

# Acknowledgement
1. https://github.com/ajaichemmanam/simple_bodypix_python for model loading function
2. https://github.com/patlevin for support functions
3. https://github.com/likeablob for download script
