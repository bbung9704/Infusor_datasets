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
    iaa.Rotate((-2,2)),
    iaa.GammaContrast((1, 1.5)),
    iaa.PiecewiseAffine(scale=(0.005, 0.01)),
    iaa.PerspectiveTransform(scale=(0.005, 0.01), keep_size=True),
], random_order=True)

for file in img_files:
    if 'aug' not in file:
        image = cv2.imread("origin/img/"+file)
        cv2.imwrite("v1/origin/img/"+file, image)
        segmap = cv2.imread("label/img/"+file.split('.')[0]+'.png')
        cv2.imwrite("v1/label/img/"+file.split('.')[0]+'.png', segmap)
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

        # Augment images and segmaps.
        for i in range(5):
            images_aug, segmaps_aug = seq(image=image, segmentation_maps=segmap)
            
            images_aug = np.array(images_aug, np.uint8)
            cv2.imwrite(f'v2/origin/img/aug_{i+1}_'+file, images_aug)
            segmaps_aug = segmaps_aug.get_arr()
            segmaps_aug = np.array(segmaps_aug, np.uint8)
            cv2.imwrite(f'v2/label/img/aug_{i+1}_'+file.split('.')[0]+'.png', segmaps_aug)

            # cat = cv2.hconcat([images_aug, segmaps_aug])
            # cv2.imshow('', cat)
            # cv2.waitKey()

        print(f"[Done] {file} aug done!")
    
    else:
        print(f"[Skip] {file} is aug file")
    

cv2.destroyAllWindows()