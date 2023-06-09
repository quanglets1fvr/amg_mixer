# [**A-Multi-Axis-Attention-MLP-Mixer-For-Biomedical-Image-Segmentation**](https://link.springer.com/book/9783031368851)

## Datasets

- Data Science Bowl 2018: 80% for training, 20 % for validation [Link data](https://www.kaggle.com/c/data-science-bowl-2018)

- Gland Segmentation [Link data](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)

- ISIC 2018 challenge: 80% for training, 10% for validation, 10% for testing [Link data](https://challenge.isic-archive.com/landing/2018/)

_All images in experiments resized to 256x256_

## Evaluation Metrics 

To evaluate our model's efficiency, we utilize Dice Coefficient (Dice) and Intersection over Union (IoU), two widely-used metrics for assessing segmentation models, especially in medical image segmentation. These metrics are capable of detecting areas in which the model requires improvement as they are sensitive to both false positive and false negative errors. The formulas for Dice and IoU are shown below:

- IOU: IOU = $\frac{TP}{TP + FP + FN}$
- Dice: Dice = $\frac{2TP}{2TP + FP + FN}$


