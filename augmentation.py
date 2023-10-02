import cv2, os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

ia.seed(3)
img_files = os.listdir("origin/img")

# Define our augmentation pipeline.
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Rotate((-3,3)),
    iaa.GammaContrast((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 2.0)),
    iaa.ElasticTransformation(alpha=(0, 1), sigma=0.25)
], random_order=True)

for file in img_files:
    if 'aug' not in file:
        image = cv2.imread("origin/img/"+file)
        segmap = cv2.imread("label/img/"+file.split('.')[0]+'.png')
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

        # Augment images and segmaps.
        for i in range(5):
            images_aug, segmaps_aug = seq(image=image, segmentation_maps=segmap)
            
            images_aug = np.array(images_aug, np.uint8)
            cv2.imwrite(f'origin/img/aug_{i+1}_'+file, images_aug)
            
            segmaps_aug = segmaps_aug.get_arr()
            segmaps_aug = np.array(segmaps_aug, np.uint8)
            cv2.imwrite(f'label/img/aug_{i+1}_'+file.split('.')[0]+'.png', segmaps_aug)
        
        print(f"[Done] {file} aug done!")
    
    else:
        print(f"[Skip] {file} is aug file")