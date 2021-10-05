# Joint Learning of Multi-level Tasks for Diabetic Retinopathy Grading on Low-resolution Fundus Images


![framework](./imgs/framework_final3.pdf)

## 1. Environment
- Python >= 3.7
- Pytorch >= 1.0 is recommended
- opencv-python
- sklearn
- matplotlib


## 2. Data Preprocess
All images are downsampled to 1024×1024, as HR images with the same resolution. Then, to obtain LR-HR pairs for training and test, all
HR images are downsampled by a factor of 4 to generate LR
images at resolution of 256 × 256.


## 3. Training and test

There are several training stages for training the DeepMT-DR model. Firstly, use the following instruction.
```
    python ./train_stage1_1.py 
```

Then, use the following instruction
```
    python ./train_stage1_2.py 
```

Finally, use the following instruction
```
    python ./train_stage2.py 
```
