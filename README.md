# This is the code for our ECCV20 Paper DGDMLSR [WIP]
----
## Requirement:
### Python 3.7 with cuda10.1 is needed
install these python packages:

>torch 1.2
>torchvision
>scikit-image
>tqdm
>opencv-python

## Data generation
### Use example image
> Step.1 use get_image.py to collect NYU depth images and depth maps.   
> Step.2 use depth_inv.py and gen_patch.py to collect patches from the image and its depth map.   
> Step.3 use gen_pkl.py to generate python pickle file to accelerate the dataloader.   

### Use your own image
> use a depth estimation model to obtain the depth map.   
> use inverse_depth to get the inversed depth   
> follow the example   

## Training
We train a specific model for each image.
> use main.py to start training, you will get the checkpoints and super resolved images for each epoch.

## Testing
>You can use the scoring script from PIRM18 to test the NIQE and PI score. You need matlab >= R2016 to process these code.