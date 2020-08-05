from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

class TestDataset(Dataset):
    def __init__(self, df, mode='test', transforms=None):
        self.df = df
        self.mode = mode
        self.transform = transforms[self.mode]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fold_number = 3
        image = Image.open(Path('/home/haykim/dataset/cv_fold{}/'.format(fold_number)) / self.df['img_file'][idx]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = self.df['class'][idx]
            
        return image, label        