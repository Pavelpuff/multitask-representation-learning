import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Callable

class GSNDataset(Dataset):
    """
    Geometric Shape Numbers dataset handling images, regression counts,
    and configuration classification targets with Joint Augmentations.
    """
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        config_to_id: dict[tuple[int, ...], int],
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.config_to_id = config_to_id

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_path).convert('L')
        counts = self.labels_df.iloc[idx, 1:].values.astype(np.float32)

        if self.is_train:
            # 1. Horizontal Flip (Right <-> Left)
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                counts[3], counts[5] = counts[5], counts[3]

            # 2. Vertical Flip (Up <-> Down)
            if torch.rand(1).item() < 0.5:
                image = TF.vflip(image)
                counts[2], counts[4] = counts[4], counts[2]

            # 3. Rotation 90 degrees with cyclic index shift
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = TF.rotate(image, angle=-90 * k)
                triangles = [counts[2], counts[3], counts[4], counts[5]]
                triangles = triangles[-k:] + triangles[:-k]
                counts[2], counts[3], counts[4], counts[5] = triangles[0], triangles[1], triangles[2], triangles[3]

        current_config = tuple(np.round(counts).astype(int))
        class_id = self.config_to_id[current_config]

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(counts, dtype=torch.float32),
            torch.tensor(class_id, dtype=torch.long)
        )