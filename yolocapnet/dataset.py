import numpy as np
import os
import torch

from torch.utils.data import Dataset, DataLoader
from utils.iou import iou_width_height as iou
from PIL import Image

# Modified from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/dataset.py

class YCNDataset(Dataset):
    def __init__(
            self, 
            train_file,
            names_file,
            data_file,
            img_dir, 
            annotations_dir,
            anchors,
            classes=6,
            S = [13, 26, 52],
            transform = None,
            image_size = 416,
    ):
        trainf = open(train_file, "r")
        self.train_file = trainf.readlines()
        trainf.close()
        self.names_file = open(names_file, "r")
        # self.data_file = open(data_file, "r")
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir

        assert(os.path.exists(img_dir), "Image directory does not exist")
        assert(os.path.exists(annotations_dir), "Annotations directory does not exist")
        assert(classes == len(self.names_file.readlines()), "Number of classes does not match number of object names")
        self.names_file.close()

        self.S = S
        self.classes = classes
        self.transform = transform
        self.image_size = image_size
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])

        self.num_anchors = self.anchors.shape[0]
        # print("Num Anchors: ", self.num_anchors)

        self.anchors_per_scale = self.num_anchors // 3
        self.ignore_iou = 0.5
    
    def __len__(self):
        return len(self.train_file)
    
    def __getitem__(self, index):
        img_path = self.img_dir + os.path.sep + self.train_file[index % len(self.train_file)].rstrip().lstrip().lstrip('data/')
        label_path = img_path.replace('.png', '.txt').replace('.jpg', '.txt')

        boxes = np.roll(np.loadtxt(label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations['image']
            boxes = augmentations['bboxes']
        
        targets = [torch.zeros((self.num_anchors, S, S, 6)) for S in self.S] # Assumes 3 scales
        for box in boxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            # print(class_label, x, y, width, height)
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                scale_index = anchor_idx // self.anchors_per_scale
                anchor_on_scale = anchor_idx % self.anchors_per_scale
                S = self.S[scale_index]
                i, j = int(S * y), int(S * x) # Cell location
                anchor_taken = targets[scale_index][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_index]:
                    targets[scale_index][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )

                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    targets[scale_index][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_index][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_index] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou:
                    targets[scale_index][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets) # x, y

# if __name__ == "__main__":
#     anchors = np.array([
#         [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#         [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#         [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
#     ])

#     dataset = YCNDataset(
#         train_file="data/train.txt",
#         names_file="data/obj.names",
#         data_file="data/obj.data",
#         img_dir="data/obj_train_data",
#         annotations_dir="data/obj_train_data",
#         anchors=anchors,
#     )

#     S = [13, 26, 52]
#     scaled_anchors = torch.tensor(anchors) / (1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
#     loader = DataLoader(dataset=dataset)

#     count =0
#     for x, y in loader:
#         boxes = []
#         count += 1
#         if count == 5: break
#         for i in range(y[0].shape[1]):
#             anchor = scaled_anchors[i]
#             print(anchor.shape)
#             print(y[i].shape)
