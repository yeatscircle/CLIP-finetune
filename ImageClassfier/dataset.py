from Config import train_tfm, test_tfm, labels_to_text, text_transformer
from torch.utils.data import Dataset
from PIL import Image
import os


class sportsDataset(Dataset):
    def __init__(self, path, transformer=test_tfm, text_transformer=text_transformer, label_to_text=labels_to_text):
        self.transformer = transformer
        self.text_transformer = text_transformer
        self.label_to_text = label_to_text
        self.labels = []
        self.images = []
        for dirpath, dirnames, filenames in os.walk(path):
            ''' 
            dirpath 是当前正在遍历的文件夹的路径
            dirnames 是当前文件夹中所有子文件夹的名字列表
            filenames 是当前文件夹中所有文件的名字列表
            '''
            for dirname in dirnames:
                dir = os.path.join(dirpath, dirname)
                for file in os.listdir(dir):
                    self.labels.append(dirname)
                    self.images.append(os.path.join(dir, file))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transformer(image)
        text = self.text_transformer(labels_to_text(self.labels[idx]))
        return text, image
