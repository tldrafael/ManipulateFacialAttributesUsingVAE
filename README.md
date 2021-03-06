# Manipulate Facial Attributes Using VAE

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1auI8GsWtazP_FHHOX0iAZ6cyC2NX3cZE?usp=sharing)

This repo presents the final project for [Computer Vision course](https://www.inf.ufsc.br/~aldo.vw/visao/) in INE / UFSC.  

The reports are written in Portuguese and saved in [reports](./reports).  

The trained model is available [here](traindir/trained_113steps/checkpoints/weights.111-1.25-1.30-1.13-0.08-0.10-0.00.h5).  

For environment compatibility, check the `environment.yml` out.  
  

## The Architecture Solution

The proposed architecture has a common VAE structure besides an additional decoder branch that predicts face masks. During training it uses face masks as labels too, which are used to replace the background of the reconstructed image such that the loss function is applied only on the face pixels. On the other hand, on the prediction mode the background replacement is made straightly by the predicted mask, not requiring any extra input but an image.  

### On training

<img src="reports/pics/VAE_Architecture_Short.png" width="800"/>

### On prediction

<img src="reports/pics/VAE_Architecture_Short_Predict.png" width="800"/>

## Playing on Colab

Test the project on Colab!! Generate spectrums or add specific attributes on your own, give a trial here. [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1auI8GsWtazP_FHHOX0iAZ6cyC2NX3cZE?usp=sharing)  

## Dataset 

This project used the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.  

Besides, as said in the [Architecture](#the-architecture-solution) session, the solution uses face masks during training. Although they are not required during prediction. For the masks extraction, I used the project [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). The masks data are available for downloading [here](https://drive.google.com/uc?id=1t85fMPgVjsWJMkT3-BP2pAFeXWg7nXpU&export=download).  

## Training

To train new models, set the CelebA dataset directory to the environment variable `celeba`. Make sure this directory contains the images and masks under the folder `imgs` and `mask_faceparsing` respectively.  
```
python training.py
```

The new training artifacts (logs and checkpoints) are stored in the `traindir` folder inside a subfolder named with training starting timestamp.  

Afterwards, the saved model need to be *converted* to correctly load on the new testing NN architecture. To guarantee it, run:  
``` 
python convert_model_toprediction.py <NEW_TRAINFILE>.h5
```
It saves a new object on the same directory with the suffix `*.predict.*` on the filename.  

## Testing

Ensure that you have the prediction version of the pretrained model:  
```
python convert_model_toprediction.py traindir/pretrained/checkpoints/weights.235-1.23-1.30-1.12-0.07-0.10-0.00.h5
```

During testing, you don't need the ground truth of the face mask, the NN generates a reconstructed version of the image and a mask in the `cache` dir.  
```
python testing.py cache/samples/077771.jpg
```

![](./reports/pics/out_077771.png)


## Adding Attribute

The present available attributs are: Bald, Bangs, Black_Hair, Blond_Hair, Eyeglasses, Gray_Hair, Heavy_Makeup, Mustache, Pale_Skin, Pointy_Nose, Smiling, Wearing_Hat, Young.  

The command bellow generates a spectrum of that attribute, and saves a file `spectrum.jpg` inside the `cache` directory.  

```
python add_attributes.py -f cache/samples/077771.jpg -a Smiling
```

![](reports/pics/out_077771_spectrum.png)

The attributes are represented by vectors stored in `cache/vector_attrs/`. You can generate more on your own.  

## Create new Vector Attributes  

To generate more vector attributes available on the celebA, run:  
```
python create_attributevectors.py Bags_Under_Eyes 
```

It creates a `cache/Bags_Under_Eyes.npy` file. Put it in the `cache/vector_attrs/` subfolder to make it available for the `add_attribute.py` script.  
