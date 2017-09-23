# ai.lock

ai.lock is a image-based authentication mechanism, alternative to biometrics. ai.lock uses deep neural networks, to extract feature images, PCA and LSH for fast and secure image matching.

### Datasets

 * [Nexus dataset](https://drive.google.com/open?id=0B-qU-nMycga7eFZiUEY3Q3V0SEU)
 * [Google Image dataset](https://drive.google.com/open?id=0B-qU-nMycga7Wm5oU2hLLUxQWk0)
 * [ALOI dataset](http://aloi.science.uva.nl/): Use 24,000 images in Illumination dataset
 * [Yfcc100m dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67) Use images with tag toy and not tag human/animal.

### Python Code

Requirements: Tensorflow, Nearpy , h5py, scikit-image, numpy, sklearn

#### Computing Inception v3 activations for a dataset of images
To create the required datasets for the experiments, take the following steps:

1. Use the Tensorflow official example code for computing the activations of a desired layer of Inception v3 network for all the images in ALOI, Google and Yfcc100m datasets. Note that, the parameters “BOTTLENECK_TENSOR_NAME” and “BOTTLENECK_TENSOR_SIZE” should be set according to the desired Inception layer: (‘pool_3/_reshape:0’ and 2048 for the last hidden layer and ‘mixed_8/pool:0’ and 49152 for Mixed8_pool0 layer). For each image in the image dataset, the code will generate a text file that contains the activation of the specified layer of Inception v3 when the image is fed as the input to the network. 
2. For each image dataset, use the *aggregate_feature_vectors.ipynb* notebook separately to aggregate the individual files generated in the previous step into a single h5py dataset. You need to set the “image_dir”, “out_file_name”, “google_images_flag” and “aloi_images_flag” parameters accordingly.
3. For each image dataset, use the *split_images.py* to split images into 5 overlapping segments (for ai.lock multi segment experiments), and run step 1 and 2 to compute the datasets corresponding to activations of the Inception v3 for each segment of the images. 
4. Arrange the dataset that are generated from steps 1-3 into the following directory hierarchy: The datasets corresponding to the last hidden layer (bottleneck_FC) and the Mixed8/pool0 layer of Inception v3 go under “bottleneck_FC” and “Mixed8_pool0” directories respectively. For each layer, put the activations corresponding to the whole size images under the “full_size” directory, and the activations corresponding to each of the 5 image segments under “part_1” to “part_5” directories. The final directory structure for the datasets should look like the following. There should be 3 files under each subdirectory, that are, Nexus.h5, ALOI.h5 and Google.h5.

Datasets

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bottleneck_FC

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;full_size

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;part_1

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;part_2

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;part_3

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;part_4

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;part_5

5. For each inception layer, run the *split_datasets_to_test_train.py* to split the embeddings in each dataset to test (holdout) and train sets. You need to set the parameter “layer_name” for bottleneck_FC and Mixed8_pool0 layers accordingly.
6. For Nexus image dataset, we had to remove 12 images as they include some sensitive personal and identifiable information. Although we cannot share the 12 images, you can download the activations of the last hidden layer in Inception v3 as well as the activations of ‘mixed_8/pool:0’ for Nexus images [here](https://drive.google.com/drive/u/2/folders/0B-qU-nMycga7ck1tX1Z2QUFQeTg?usp=sharing).
7. Download the Nearpy package. Then, replace the last line of hash_vector method in hashes/permutations/randombinaryprojections.py (i.e. return [''.join(['1' if x > 0.0 else '0' for x in projection])]) to the following code:

```python
projection[projection > 0] = 1
projection[projection <= 0] = 0
return projection
```

#### Running Single/Multi Layer Single Image experiments
Use the code provided under Single_image directory. To compute the best performing thresholds for binary classifications of images, use *single_image_cv_train.py*.
To evaluate the performance of ai.lock on holdout set use *single_image_cv_holdout.py*.

#### Running Single/Multi Layer Multi Image experiments
Use the code provided under Multi_image directory. To compute the best performing thresholds for binary classifications of images, and evaluate the performance of ai.lock on holdout set use multi_image.py.

#### Synthetic image attack 

We have adapted the image augmentation code of [Sean Huver](https://github.com/huvers/img_augmentation).
You can find the modified code in *augment.py*. 
 
We used the official [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow) Tensorflow implementation to train the DCGAN network on Nexus images and generate synthetic images.
