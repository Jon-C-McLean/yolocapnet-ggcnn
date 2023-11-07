import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import numpy as np

NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416

NUM_CLASSES=6
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45

S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_FILE = "ycn_checkpoint.pth.tar"
IMG_DIR = ""
LABEL_DIR = ""

ANCHORS = np.array([
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
])  # Note these have been rescaled to be between [0, 1]


scale = 1.1

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE*scale),
            min_width=int(IMAGE_SIZE*scale),
            border_mode = cv.BORDER_CONSTANT,
        ),
        A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.3),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=1.0, border_mode=cv.BORDER_CONSTANT
                ),
                # A.IAAAffine(
                #     shear=15, p=0.5, mode="constant"
                # ),
            ],
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.01),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

OBJ_NAME_PATH = "data/obj.names"
CLASSES=map(lambda x: x.rstrip().lstrip(), open(OBJ_NAME_PATH).readlines())

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tqdm import tqdm
from yolo import YOLOv3
from yolo_sample import YOLOv3 as YOLO_Sample
from loss import YOLOLoss
from dataset import YCNDataset, DataLoader
from utils.checkpoint import save_checkpoint

device = torch.device('mps' if torch.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def train_fn(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(loader, leave=True)

    history = {
        'loss': [],
    }

    for batch_index, (x,y) in enumerate(loop):
        x = x.to(device)
        # print(x.shape)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        # scaled_anchors.to(device)

        out = model(x)
        loss = (loss_fn(out[0], y0, scaled_anchors[0]) + loss_fn(out[1], y1, scaled_anchors[1]) + loss_fn(out[2], y2, scaled_anchors[2])) 
        
        history['loss'].append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

        mean_loss = sum(history['loss']) / len(history['loss'])
        loop.set_postfix(loss=mean_loss)

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(device)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()

def main():
    # model = YOLOv3(num_class=NUM_CLASSES).to(device)
    model = YOLO_Sample(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = YCNDataset(
        train_file="/Users/jon/development/capstone/yolo_initial/train.txt",
        names_file="/Users/jon/development/capstone/yolo_initial/obj.names",
        data_file="/Users/jon/development/capstone/yolo_initial/obj.data",
        img_dir="/Users/jon/development/capstone/yolo_initial/",
        annotations_dir="/Users/jon/development/capstone/yolo_initial/",
        anchors = ANCHORS,
        classes=NUM_CLASSES,
        transform=train_transforms,
    )

    print(torch.tensor(ANCHORS).shape)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,
        # pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(torch.float32).to(device)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 3 == 0:
            # check_class_accuracy(model, train_loader, threshold=0.5)
            model.train()
        
    if SAVE_MODEL:
        save_checkpoint(model, optimizer, filename=CHECKPOINT_FILE)

if __name__ == "__main__":
    main()