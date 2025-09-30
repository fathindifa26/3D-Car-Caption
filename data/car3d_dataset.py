import os
import json
from torch.utils.data import Dataset
from PIL import Image
from data.utils import pre_caption

class car3d_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        Dataset for car 3D caption training
        image_root (string): Root directory of images (e.g. datasets/car3d/images/)
        ann_root (string): directory containing annotation files
        '''        
        filename = 'train.json'
        
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        # Create image ID mapping untuk compatibility dengan BLIP
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            # Extract image ID dari filename (misalnya img_0001 dari img_0001_p-30_y0.jpg)
            image_name = ann['image']
            img_id = image_name.split('_')[1]  # ambil angka setelah img_
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
        print(f"Loaded {len(self.annotation)} training samples")
        print(f"Unique images: {len(self.img_ids)}")
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        
        # Load image
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        # Process caption dengan prompt
        caption = self.prompt + pre_caption(ann['caption'], self.max_words) 

        # Get image ID untuk compatibility
        img_id = ann['image'].split('_')[1]  # Extract ID from filename
        
        return image, caption, self.img_ids[img_id]


class car3d_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split='val'):  
        '''
        Dataset for car 3D caption evaluation
        split (string): 'val' or 'test'
        '''
        filename = f'{split}.json'
        
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        
        print(f"Loaded {len(self.annotation)} {split} samples")
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        
        # Load image
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        # Extract image ID dari filename
        img_id = ann['image'].split('_')[1]  # img_0001 -> 0001
        
        return image, int(img_id)


class car3d_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split='val', max_words=30):  
        '''
        Dataset for car 3D image-text retrieval evaluation
        '''
        filename = f'{split}.json'
        
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        
        # Prepare data structures untuk retrieval
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        # Group captions by unique image
        image_groups = {}
        for ann in self.annotation:
            img_name = ann['image']
            if img_name not in image_groups:
                image_groups[img_name] = []
            image_groups[img_name].append(ann['caption'])
        
        # Build retrieval mappings
        txt_id = 0
        for img_id, (img_name, captions) in enumerate(image_groups.items()):
            self.image.append(img_name)
            self.img2txt[img_id] = []
            
            for caption in captions:
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.image[index])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index