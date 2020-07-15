# Lesson 1

You usually want to do something like 
```
from fastai.<FUNCTIONALITY> import *
```
at the start of most projects. 

Lesson 1 discuss something called "fine grained image analysis" which means trying to distinguish between similar looking images, in this case dog and cat breeds. There are some helpful functions to download and prepare the images which you can see within the notebook.

## Training the Model 

Once you have your set of labeled images, the first thing you need to do is normalize them. For example using a command like this.

```
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
```
In this case all images are cropped to 224. The reason we do this is due to GPU limitations. In order to parallelize the learning processes on the GPU it needs to be able to operate on similarly sized things. You can see this operation returns an `ImageDataBunch`, this will automatically split up the data for you into a training and validation set, among other things.

The model we will be using for a base is **resnet34** which is the convocational neural net architecture, resnet has also been pertained for recognizing images so it has some preset weights, this means we start with a model that is already "good" at classifying images, this reduces training time. The other popular one is **resnet50** which has more layers but a large increase in training time. For most uses resnet34 seems adequate. 

```
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
```

And now that we have a learner we can, well, make it learn.
```
learn.fit_one_cycle(4)  # 4 means do 4 epochs
```

Now we have a trained model, and we get an error rate. One thing we may want to do is look at what we got wrong. 
```
interp.plot_top_losses(9, figsize=(15,11))
```

This will show us a set of images and our incorrect guesses. Another way to visualize this is with something called a confusion matrix. It shows us categories that we often mix up.

```
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```