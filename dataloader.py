from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from sklearn.model_selection import StratifiedKFold

IMAGES_DIRS = '../Competition/Jungle/train_features/'
TRAIN_LABELS = 'data/train_annotations.csv'


class WhaleDataset(Dataset):
    def __init__(self, path, image_ids, labels, transform):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = str(self.image_ids[item])
        img = cv2.imread(self.path + image_id + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)
        image = img["image"]
        label = self.labels[item]

        return {
            "x": image,
            "y": label,
        }


class WhaleDataModule(pl.LightningDataModule):
    def __init__(self, img_size, batch_size):
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            # A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.25), shear=(-3, 3),
            #          p=0.5),
            # A.GaussianBlur(blur_limit=(3, 7), p=0.05),
            # A.GaussNoise(p=0.05),
            # A.RandomBrightnessContrast(p=0.5),
            # A.HorizontalFlip(p=0.5),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()])
        self.test_transform = A.Compose([A.Resize(self.img_size, self.img_size),
                                         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ToTensorV2()])
        self.hparams['transform'] = self.train_transform
        self.save_hyperparameters()

    def prepare_data(self):
        # Run StratifiedKFold
        # df = pd.read_csv(TRAIN_LABELS)
        # df["kfold"] = -1
        # df = df.sample(frac=1).reset_index(drop=True)
        # stratify = StratifiedKFold(n_splits=5)
        # for i, (t_idx, v_idx) in enumerate(stratify.split(X=df.id.values, y=df.label.values)):
        #     df.loc[v_idx, "kfold"] = i
        # df.to_csv('data/train_folds.csv', index=False)
        pass

    def setup(self, stage=None):
        dfx = pd.read_csv('data/train_folds.csv')
        train = dfx.loc[dfx["kfold"] != 1]
        val = dfx.loc[dfx["kfold"] == 1]
        self.train_dataset = WhaleDataset(IMAGES_DIRS,
                                          image_ids=train.id.values,
                                          labels=train.label.values,
                                          transform=self.train_transform)
        self.valid_dataset = WhaleDataset(IMAGES_DIRS,
                                          image_ids=val.id.values,
                                          labels=val.label.values,
                                          transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=4)

    # def test_dataloader(self):
    # pass


class WhaleTestDataset(Dataset):
    def __init__(self, path, image_ids, img_size):
        super().__init__()
        self.path = path
        self.image_ids = image_ids
        self.img_size = img_size
        self.transform = A.Compose([A.Resize(self.img_size, self.img_size),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ToTensorV2()])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = str(self.image_ids[item])
        image = cv2.imread(self.path + image_id + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        return {
            "x": image["image"],
        }
