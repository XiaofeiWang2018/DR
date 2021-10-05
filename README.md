# Deep Multi-Task Learning for Diabetic Retinopathy Grading in Fundus Images
- This is the official repository of the paper "Deep Multi-Task Learning for Diabetic Retinopathy Grading in Fundus Images
" from **AAAI 2021**[[Paper Link]](https://www.aaai.org/AAAI21Papers/AAAI-2900.WangX.pdf, "Paper Link")

![framework](./imgs/framework_final.jpg)

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


## 3. Training
The details of the hyper-parameters are all listed in the `train.py`. Use the below command to train our model on the SIGF database.

```
    python ./train.py 
```

## 4. Test
Download the pre-trained model in [[Dropbox]](https://www.dropbox.com/s/e1oebawbp5wlpvm/pretrained_model.zip?dl=0). Then put the file in tghe directory of 
`pretrained_model`. Use the below command to test the model on the SIGF database.
```
    python ./test.py 
```

## 5. Compared Methods

The network re-implenmentation of [[Chen et al.]](https://ieeexplore.ieee.org/abstract/document/7318462/, "Chen") is in the file of:
`chen_net.py`
and from the directory of `./Compared Methods`




## 6. Ablation Study

If you are interested in our ablation study, please see `./Ablation study`




## 7. Network Interpretability

1. If you are interested in the visualization method and results used for showing the interpretability 
of our method, please refer to the directory of `./saliency`



2. Or you can just see the images in the directory of `./visualization_result`
for more visualization results. Some examples of the visualization rsults are shown here.

![Database](./imgs/figure1.jpg)


## 8. Citation
If you find our work useful in your research or publication, please cite our work:
```
@article{Li2020deep,
  title={DeepGF: Glaucoma Forecast Using the Sequential Fundus Images.},
  author={Li, Liu and Wang, Xiaofei and  Xu, Mai and Liu, Hanruo},
  journal={MICCAI},
  year={2020}
}
```
