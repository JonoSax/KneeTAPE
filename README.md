# Purpose

To observe a user performing knee flexion exercises and track their motion and calculate their range of motion across the entirety of the video

# Getting Started
1. Install requirements in virtual environment

```bash
python -m venv venvSBP
. venvSBP/bin/activate
pip install -r requirements.txt
```

2. Download desired tfjs models for bodypix.
```bash
# For example, download a ResNet50-based model to ./bodypix_resnet50_float_model-stride16
# bodypix/mobilenet/float/050/model-stride8.json
$ ./get-model.sh bodypix/resnet50/float/model-stride16
```

3. Set path to models and image for inference in .py files
```py
videoPath = './awesome_vid.mov'
modelPath = './bodypix_resnet50_float_model-stride16/'
```
4. Run the script
```bash
python evalbody_singleposemodelVIDEO.py
```

# Acknowledgement
1. https://github.com/ajaichemmanam/simple_bodypix_python for model loading function
2. https://github.com/patlevin for support functions
3. https://github.com/likeablob for download script
