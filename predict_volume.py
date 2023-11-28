import cv2, os
import numpy as np

def cal_iou(tgt_np, compare):
    overlap = tgt_np * compare
    union = tgt_np + compare
    iou = overlap.sum()/float(union.sum())
    return iou

result = [0 for _ in range(23)]
cnt = [0 for _ in range(23)]

tgt = cv2.imread('dataset/label/img/try_0_cap_90.png')
tgt_np = np.array(tgt, dtype=bool)

root = 'dataset/label/img'
labels = os.listdir(root)
for label in labels:
    file_name = label.split('.')[0]
    volume = file_name.split('_')[-1]
    idx = int(int(volume) / 10)

    img = cv2.imread(root + '/' + label)
    compare = np.array(img, dtype=bool)
    iou = cal_iou(tgt_np, compare)

    cnt[idx] += 1
    result[idx] += iou

for i in range(len(result)):
    result[i] /= cnt[i]

max_iou = max(result)
predict = result.index(max_iou)
print(result)
print(max_iou)
print(predict * 10)