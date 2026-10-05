import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import both_transforms, highres_transform, lowres_transform


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        entries = sorted(os.listdir(root_dir))
        if entries and all(os.path.isdir(os.path.join(root_dir, item)) for item in entries):
            self.class_names = entries
            for index, name in enumerate(self.class_names):
                files = sorted(os.listdir(os.path.join(root_dir, name)))
                self.data += list(zip(files, [index] * len(files)))
        else:
            self.class_names = ["default"]
            self.data = [(file_name, 0) for file_name in sorted(entries) if os.path.isfile(os.path.join(root_dir, file_name))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label]) if len(self.class_names) > 1 else self.root_dir

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = both_transforms(image=image)["image"]
        high_res = highres_transform(image=image)["image"]
        low_res = lowres_transform(image=image)["image"]
        return low_res, high_res


def test_dataset():
    dataset_root = "new_data/" if os.path.exists("new_data/") else "images"
    if os.path.exists(dataset_root):
        dataset = MyImageFolder(root_dir=dataset_root)
        loader = DataLoader(dataset, batch_size=1, num_workers=0)
        for low_res, high_res in loader:
            print("Dataset check successful!")
            print("Low-res shape:", low_res.shape)
            print("High-res shape:", high_res.shape)
            break
    else:
        print("Dataset directory not found. Dataset test skipped.")


if __name__ == "__main__":
    test_dataset()