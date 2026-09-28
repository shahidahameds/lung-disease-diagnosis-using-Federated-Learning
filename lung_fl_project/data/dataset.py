import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import DATA_DIR, IMAGE_EXTENSIONS, RESULTS_DIR


def _image_files(directory):
    valid_extensions = {extension.lower() for extension in IMAGE_EXTENSIONS}
    paths = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def discover_class_mapping(data_dir=DATA_DIR):
    classes = set()
    for split in ("train", "test", "val"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        classes.update(
            name for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
        )
    if not classes:
        raise FileNotFoundError(f"No class directories found in '{data_dir}'.")
    class_to_idx = {name: index for index, name in enumerate(sorted(classes))}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mapping_path = os.path.join(RESULTS_DIR, "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as mapping_file:
        json.dump(class_to_idx, mapping_file, indent=2, sort_keys=True)
    return class_to_idx


def build_dataset_report(data_dir=DATA_DIR):
    class_to_idx = discover_class_mapping(data_dir)
    report = {
        "detected_classes": list(class_to_idx),
        "training_images_per_class": {},
        "test_images_per_class": {},
        "total_training_images": 0,
        "total_test_images": 0,
        "image_dimensions": {},
    }
    for split, output_key in (("train", "training_images_per_class"), ("test", "test_images_per_class")):
        for class_name in class_to_idx:
            paths = _image_files(os.path.join(data_dir, split, class_name))
            report[output_key][class_name] = len(paths)
            total_key = "total_training_images" if split == "train" else "total_test_images"
            report[total_key] += len(paths)
            for path in paths:
                with Image.open(path) as image:
                    dimension = f"{image.width}x{image.height}"
                report["image_dimensions"][dimension] = report["image_dimensions"].get(dimension, 0) + 1
    report_path = os.path.join(RESULTS_DIR, "dataset_report.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved dataset report to: {report_path}")
    return report

class LungDiseaseDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=(224, 224), augment=False):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self._tensor_cache = {} if not augment else None
        
        # Discover one mapping across all splits so train and test labels agree.
        split_dir = os.path.join(self.data_dir, split)
        self.class_to_idx = discover_class_mapping(self.data_dir)
        self.classes = list(self.class_to_idx)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Dataset split '{split}' was not found at '{split_dir}'.")
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            self.samples.extend((path, self.class_to_idx[cls_name]) for path in _image_files(cls_dir))
                    
        # Define preprocessing
        if self.augment and self.split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if self._tensor_cache is not None and idx in self._tensor_cache:
            return self._tensor_cache[idx], label
        image = Image.open(img_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
        if self._tensor_cache is not None:
            self._tensor_cache[idx] = image
            
        return image, label
