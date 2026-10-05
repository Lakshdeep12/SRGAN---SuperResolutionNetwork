import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = both_transforms(image=image)["image"]
        high_res = highres_transform(image=image)["image"]
        low_res = lowres_transform(image=image)["image"]
        return low_res, high_res

def test_dataset():
    if os.path.exists("new_data/"):
        dataset = MyImageFolder(root_dir="new_data/")
        loader = DataLoader(dataset, batch_size=1, num_workers=2)
        for low_res, high_res in loader:
            print("Dataset check successful!")
            print("Low-res shape:", low_res.shape)
            print("High-res shape:", high_res.shape)
            break
    else:
        print("'new_data/' directory not found. Dataset test skipped.")

if __name__ == "__main__":
    test_dataset()