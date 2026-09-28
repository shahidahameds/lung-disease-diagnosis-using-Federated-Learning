import os
import glob
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from config import DATA_DIR, RESULTS_DIR, IMAGE_EXTENSIONS, TOTAL_CLIENTS, TEST_SPLIT
from data.dataset import build_dataset_report

def inspect_dataset():
    """
    Inspects the DATA_DIR, determines class distribution, creates FL partitions, 
    and saves the summary and plot.
    """
    print("\n--- PHASE 1: Dataset Inspection ---")
    build_dataset_report(DATA_DIR)
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    
    has_predefined_split = os.path.exists(train_dir) and os.path.exists(test_dir)
    classes = {}
    
    if has_predefined_split:
        print("Detected predefined train/test split.")
        for split in ['train', 'test']:
            split_dir = os.path.join(DATA_DIR, split)
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir): continue
                
                images = []
                for ext in IMAGE_EXTENSIONS:
                    images.extend(glob.glob(os.path.join(class_dir, f"*{ext}")))
                    images.extend(glob.glob(os.path.join(class_dir, f"*{ext.upper()}")))
                    
                if class_name not in classes:
                    classes[class_name] = {'train': [], 'test': []}
                classes[class_name][split] = images
    else:
        print("No predefined split found. Will split automatically based on TEST_SPLIT.")
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        for class_name in os.listdir(DATA_DIR):
            class_dir = os.path.join(DATA_DIR, class_name)
            if not os.path.isdir(class_dir): continue
            
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(glob.glob(os.path.join(class_dir, f"*{ext}")))
                images.extend(glob.glob(os.path.join(class_dir, f"*{ext.upper()}")))
                
            # Sort for deterministic splitting before shuffle
            images.sort()
            
            random.seed(42)
            random.shuffle(images)
            split_idx = int(len(images) * (1 - TEST_SPLIT))
            
            classes[class_name] = {
                'train': images[:split_idx],
                'test': images[split_idx:]
            }

    total_images = sum(len(c.get('train', [])) + len(c.get('test', [])) for c in classes.values())
    if total_images == 0:
        print(f"ERROR: No images found in '{DATA_DIR}'!")
        print("Please place the lung disease image dataset in the 'data/' folder.")
        print("Format: data/<class_name>/<images>")
        return False
        
    print(f"Found {total_images} images across {len(classes)} classes: {list(classes.keys())}")
    
    # Extract dimensions from a sample image
    sample_img_path = None
    for c_data in classes.values():
        if len(c_data['train']) > 0:
            sample_img_path = c_data['train'][0]
            break
            
    if sample_img_path:
        with Image.open(sample_img_path) as img:
            print(f"Sample image dimension: {img.size}")
    
    # Create partitions summary
    summary_data = []
    for class_name, data in classes.items():
        train_imgs = data['train']
        test_imgs = data['test']
        num_train = len(train_imgs)
        imgs_per_dept = math.floor(num_train / TOTAL_CLIENTS)
        
        summary_data.append({
            'class': class_name,
            'total_images': num_train + len(test_imgs),
            'training_images': num_train,
            'test_images': len(test_imgs),
            'images_per_virtual_department': imgs_per_dept
        })
        
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(RESULTS_DIR, "dataset_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset summary to: {csv_path}")
    
    # Generate class-distribution plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['class'], df['total_images'], color='skyblue')
    plt.title('Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(plot_path)
    print(f"Saved class distribution plot to: {plot_path}")
    
    return True
