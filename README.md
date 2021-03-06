# Semantic Segmentation

### Introduction
This project uses a Fully Convolutional Neural Network (FCN), implemented in TensorFlow, to detect the road surface in images taken from the perspective of a driver in a car.  The process of detecting areas of an image belonging to a certain category such as road, vehicle, pedestrian, etc. is called <i>semantic segmentation</i>.  This is to be distinguished from merely identifying an object with a bounding box.  In semantic segmentation, the actual shape of the object is identified pixel by pixel.  See example image below.

<br/>
<p align="center"> <span> <img width="280px" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21.jpg" alt="biker"> </span> &nbsp;&nbsp;&nbsp; <span> <img width="280px" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_class.png" alt="biker"> </span> <br> <small><i>Left</i>: Input image. <i>Right</i>: It's semantic segmentation. <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html">Source.</a> (Courtesy of http://blog.qure.ai)</small></p>
<br />


### VGG16 Pre Trained Model
The encoder for the FCN is the VGG16 pretrained model.  VGG16 is a 16 layer CNN developed by Visual Geometry Group and trained on the ImageNet dataset for image recognition.  It can recognize over 1000 distinct images.  The VGG16 pre-trained model is read in but only the layers leading up to, but not including, the first fully connected layer are used as the encoder of the FCN.  

<br /><br />
<p align="center">
<img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/08131808/temp7.png" width="470px" /><br /><b>VGG16 Architecture (Courtesy www.analyticsvidhya.com)</b></p>
<br />


### Complete FCN Architecture
The decoder for the FCN starts with a 1x1 convolution of layer 7 of VGG16.  This is then upsampled with a 4x4 kernel and stride of 2 thereby doubling the width and height of the output.  VGG16 layer 4 is then added as a skip layer.  This process of upsampling and skip layer is repeated with VGG16 layer 3, again doubling the dimensions.  Finally, the result is further upsampled with a 16x16 kernel and  stride of 8, thus multiplying the height and width by 8.  These are the final dimensions of the FCN and the same dimensions as the original input image.  Each original pixel can now represented by logits with a depth of 2, indicating the likelihood that the pixel is road or not road.  These logits are run through a softmax to turn them into probabilities.  This allows us to edit the original image by coloring all pixels identified as road in a chosen color - green in this case.


### Data
The FCN is trained on the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).  This consists of 289 training and 290 test images of drive scenes in various scenarios. There are three basic types of road images:

* uu - urban unmarked (98/100)
* um - urban marked (95/96)
* umm - urban multiple marked lanes (96/94)

The road segments of all of these images have already been pre-labeled by hand.  

<br /><br />
<p align="center">
<img src="http://www.cvlibs.net/datasets/kitti/images/header_road.jpg"  /><br /><b>Typical Images (Courtesy http://www.cvlibs.net)</b></p>
<br />


### Training
The outputs from the model are measured against the labeled data using a cross entropy operation.  This cost function is then optimized per batch using an Adam optimizer.  

The available hyper parameters for possible tuning are number of epochs, batch size, learning rate, keep probability, and L2 regularization scale for each of the layers.  

I maintained keep probability at 0.5 and all L2 regularization scales at 0.001 so these were not actually tuned.

This left the number of epochs, batch size, and learning rate as the actual tunable parameters.

After some experimentation with just 2 epochs, I came to realize that the batch size could go very small, gaining a great deal of training out of each epoch.  I have no good explanation for this.

Starting with a learning rate of 0.001 I kept pushing it smaller and smaller as I realized I needed to get my loss to converge on a smaller number less than 0.1.  This may seem like a large loss value but realize that this loss figure is not just the cross entropy but also includes the L2 regularization from six layers!  I ended up with learning rates in range of 0.0002 and 0.0001 in order to converge adequately on the this level of total loss.

### Results

My best results were with Epochs = 50, batch size = 2, learning rate = 0.0002.  The total loss (cross entropy and all regularizations) was 0.092.  Below are a few of the better identifications of road segments in the test images.

<br /><br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/semantic_segmentation/blob/master/images/um_000022.png"  /><br /><b>um_000022.png</b></p>
<br />

<br /><br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/semantic_segmentation/blob/master/images/umm_000009.png"  /><br /><b>umm_000009.png</b></p>
<br />

<br /><br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/semantic_segmentation/blob/master/images/uu_000010.png"  /><br /><b>um_000022.pnguu_000010.png</b></p>
<br />





### Comments

I wanted to experiment with some things but felt a bit restricted by the shape of the functions and unit tests which were part of the requirements of the project.  In particular, I wanted to experiment with my own declining learning rate function which I have done in the past.  I also wanted to experiment modifying batch size and learning rate over epochs.  For example, run 5 epochs with small batch size and aggresive learning rate, then run 10 epochs with larger batch size and smaller learning rate, etc.  This would require tampering with the signatures of the functions under test so I didn't do this.

<hr>
<hr>
# Semantic Segmentation

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
