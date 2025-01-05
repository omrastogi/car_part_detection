Dataset Preparation and Augmentation:
-------------------------------------
The dataset consists of car images, classified into seven categories:
- Front
- Front-Left
- Front-Right
- Rear
- Rear-Left
- Rear-Right
- None (for ambiguous or irrelevant images)

Annotation to Class Mapping:
The annotations were mapped to class using the following map:
| Class         | Annotations                                                                                         |
|---------------|-----------------------------------------------------------------------------------------------------|
| Front         | bonnet, frontbumper, frontws, headlightwasher, indicator, leftheadlamp, rightheadlamp,              |
|               | frontbumpergrille, lowerbumpergrille, licenseplate, namebadge                                       |
| Rear          | rearbumper, rearws, fuelcap, taillamp, rearbumpercladding, leftbootlamp, rightbootlamp,             |
|               | towbarcover, lefttailgate, righttailgate, rearbumpermissing, rearwsmissing                          |
| Front-Right   | rightfender, rightfrontdoor, rightfrontdoorglass, rightorvm, rightfoglamp,                          |
|               | partial_rightfender, partial_rightfrontdoor, rightfrontbumper                                       |
| Front-Left    | leftfender, leftfrontdoor, leftfrontdoorglass, leftorvm, leftfoglamp, partial_leftfender,           |
|               | partial_leftfrontdoor, leftfrontbumper                                                              |
| Rear-Right    | rightqpanel, rightreardoor, rightreardoorglass, rightrearventglass, partial_rightqpanel,            |
|               | partial_rightreardoor, rightrearbumper                                                              |
| Rear-Left     | leftqpanel, leftreardoor, leftreardoorglass, leftrearventglass, partial_leftqpanel,                 |
|               | partial_leftreardoor, leftrearbumper                                                                |
| None          | alloywheel, antenna, car, cracked, dirt, logo, reflection, rust, scratch, shattered, sensor,        |
|               | sunroof, wiper, series                                                                              |

Final Label Selection:
For each image, the area of each class was calculated by summing the areas of their respective annotations (compositions). 
Afterward, the class with the largest cumulative area was selected as the label for the image. 
There was an extra threshold for the 'None' class as it was all the images, and occupied large area in several images.

Train/val/test Split:
The dataset was split into training, validation, and testing sets using a stratified split to maintain class balance. 
Training: 3656 images  
Validation: 159 images  
Testing: 159 images 

This split, despite a 5% variability, represents a justified tradeoff to achieve a noticeable improvement in the F1 score, aiming to boost it from 82% to 86%.


Preprocessing:
- Images were resized to 380x380 resolution.
- Normalization was applied using standard ImageNet mean and standard deviation.

Augmentation (applied only to the training dataset):
- Random horizontal flipping (excluding certain labels to preserve viewpoint integrity).
- Random affine transformations (rotation up to 45° and scaling between 0.7x and 1.3x).
- Random resized cropping (scale between 0.8x and 1.2x of the image).
- Color jitter for brightness, contrast, saturation, and hue adjustments.

Model Used:
-----------
The model used is a ConvNeXt-Tiny architecture, pre-trained on ImageNet, and fine-tuned for classifying the car viewpoints. 
- Number of Parameters: ~28.6M
- Input Image Size: 380x380

Training Parameters:
--------------------
- Optimizer: AdamW (learning rate = 5e-4, weight decay = 1e-2)
- Loss Function: CrossEntropyLoss
- Learning Rate Scheduler: CosineAnnealingLR with `T_max = 5000` iterations and `eta_min = 1e-6`
- Number of Iterations: 10,000
- Gradient Accumulation: 2 
- Progressive Unfreezing with interval: 400 
- Label Smoothing: 0.1 

Hyper-parameters:
- Data Augmentation: Enabled for the training set as described above.
- Validation Interval: Every 50 iterations.
- Checkpoint Interval: Every 1,000 iterations.

Optimized Model:
----------------
The trained ConvNeXt model was converted to TensorFlow Lite format using ai_edge_torch for efficient inference on edge devices. The exported `.tflite` model is available for download and evaluation.
