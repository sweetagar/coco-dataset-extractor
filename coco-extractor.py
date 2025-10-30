from pycocotools.coco import COCO # pip install pycocotools
import requests
import os
import sys
import threading
import queue
import argparse

# Global counters for progress tracking
train_counter = 0
val_counter = 0
counter_lock = threading.Lock()

def makeDirectory(dirName):
    try:
        os.mkdir(dirName)
        print(f"Made {dirName} Directory.")
    except:
        pass

def count_existing_files(className, split_name):
    """Count already downloaded files"""
    actual_split = "valid" if split_name == "val" else split_name
    image_dir = f'{className}_dataset/{actual_split}/images'
    if os.path.exists(image_dir):
        return len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    return 0

def download_worker(image_queue, className, split_name, coco_instance, catIds, total_images):
    """Worker thread that downloads images from queue"""
    global train_counter, val_counter, counter_lock
    
    while True:
        try:
            im = image_queue.get(timeout=1)  # Get next image from queue
            image_file_name = im['file_name']
            label_file_name = im['file_name'].split('.')[0] + '.txt'

            actual_split = "valid" if split_name == "val" else split_name
            image_path = f'{className}_dataset/{actual_split}/images/{image_file_name}'
            label_path = f'{className}_dataset/{actual_split}/labels/{label_file_name}'
            
            fileExists = os.path.exists(image_path)
            if(fileExists):
                print(f"{className}. {image_file_name} - To be Downloaded again ({split_name})")
                os.remove(image_path)
                if os.path.exists(label_path): os.remove(label_path)
                
                # Decrease counter when deleting existing file
                with counter_lock:
                    if split_name == "train":
                        train_counter -= 1
                    else:
                        val_counter -= 1

            img_data = requests.get(im['coco_url']).content
            annIds = coco_instance.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
            anns = coco_instance.loadAnns(annIds)    
            
            # Update counter and show progress
            with counter_lock:
                if split_name == "train":
                    train_counter += 1
                    current = train_counter
                else:
                    val_counter += 1  
                    current = val_counter
            dl_str = 'Downloading' if not fileExists else 'Re-downloading'
            print(f"{className}. {dl_str} - {image_file_name} ({split_name} {current}/{total_images})")
            
            # Create YOLO format annotation
            label_content = ""
            for i in range(len(anns)):
                topLeftX = anns[i]['bbox'][0] / im['width']
                topLeftY = anns[i]['bbox'][1] / im['height']
                width = anns[i]['bbox'][2] / im['width']
                height = anns[i]['bbox'][3] / im['height']
                
                s = "0 " + str((topLeftX + (topLeftX + width)) / 2) + " " + \
                str((topLeftY + (topLeftY + height)) / 2) + " " + \
                str(width) + " " + \
                str(height)
                
                if(i < len(anns) - 1):
                    s += '\n'
                label_content += s
            
            with open(image_path, 'wb') as image_handler:
                image_handler.write(img_data)
            with open(label_path, 'w') as label_handler:
                label_handler.write(label_content)       
            image_queue.task_done()  # Mark task as completed
            
        except queue.Empty:
            break  # No more images in queue

def getImagesFromClassName(className, coco_instance, split_name, num_workers=4):
    actual_split = "valid" if split_name == "val" else split_name
    makeDirectory(f'{className}_dataset/{actual_split}/images')
    makeDirectory(f'{className}_dataset/{actual_split}/labels')
    catIds = coco_instance.getCatIds(catNms=[className])
    imgIds = coco_instance.getImgIds(catIds=catIds )
    images = coco_instance.loadImgs(imgIds)
    total_images = len(images)

    # Initialize counter and safe resume
    existing_count = count_existing_files(className, split_name)
    safe_start = max(0, existing_count - max(4, num_workers))
    
    global train_counter, val_counter
    if split_name == "train":
        train_counter = existing_count
    else:
        val_counter = existing_count

    print(f"Total Images: {total_images} for class '{className}' in {split_name} set")
    print(f"Existing files: {existing_count}, resuming from: {safe_start}")
    print(f"Starting {num_workers} download workers...")

    # Create queue and add images from safe start point
    image_queue = queue.Queue()
    for im in images[safe_start:]:
        image_queue.put(im)

    # Start worker threads
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=download_worker, 
                           args=(image_queue, className, split_name, coco_instance, catIds, total_images))
        threads.append(t)
        t.start()

    # Wait for all downloads to complete
    for t in threads:
        t.join()
        
    print(f"Completed downloading for {className} {split_name} set")

def create_data_yaml(className):
    yaml_content = f"""path: {className}_dataset
train: train/images
val: valid/images

nc: 1
names: ['{className}']
"""
    
    with open(f'{className}_dataset/data.yaml', 'w') as f:
        f.write(yaml_content)
    print(f"Created data.yaml for {className}")

def format_size(bytes_size):
    """Convert bytes to human readable format"""
    if bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f}MB"

def count_class_images_with_size(className, coco_instance, split_name):
    """Simple count for one class in one split with size estimate"""
    catIds = coco_instance.getCatIds(catNms=[className])
    if not catIds:
        return 0, 0
    imgIds = coco_instance.getImgIds(catIds=catIds)
    images = coco_instance.loadImgs(imgIds)
    
    # Estimate size from first 10 images (for speed)
    sample_size = min(10, len(images))
    total_size = 0
    for im in images[:sample_size]:
        # Rough estimate: width * height * 3 bytes / 8 (JPEG compression)
        estimated_size = (im['width'] * im['height'] * 3) // 8
        total_size += estimated_size
    
    if sample_size > 0:
        avg_size = total_size / sample_size
        total_estimated_size = avg_size * len(images)
    else:
        total_estimated_size = 0
        
    return len(images), total_estimated_size

def list_all_classes_with_size(train_coco, val_coco):
    """Simple list of all classes with counts and sizes"""
    cats = train_coco.loadCats(train_coco.getCatIds())
    print("COCO Classes:")
    for cat in sorted(cats, key=lambda x: x['name']):
        train_count, train_size = count_class_images_with_size(cat['name'], train_coco, "train")
        val_count, val_size = count_class_images_with_size(cat['name'], val_coco, "val")
        total_size_mb = (train_size + val_size) / (1024 * 1024) * 1.5  # Rough estimate with compression factor
        print(f"  {cat['name']}: {train_count} train, {val_count} val, ~{total_size_mb:.1f}MB")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download COCO dataset for specific classes")
    parser.add_argument("classes", nargs="*", help="Class names to download (e.g., horse dog cat)")
    parser.add_argument("--worker", type=int, default=4, help="Number of download workers (default: 4)")
    parser.add_argument("--countonly", action="store_true", help="Count images and estimate size for specified classes only (no download)")
    parser.add_argument("--countall", action="store_true", help="List all COCO classes with image counts and sizes")
    return parser.parse_args()

# Parse arguments
args = parse_arguments()
classes = [cls.lower() for cls in args.classes] if args.classes else []
num_workers = args.worker

# Handle count-only arguments
if args.countall or args.countonly:
    print("Loading COCO annotations...")
    train_coco = COCO('annotations/instances_train2017.json')
    val_coco = COCO('annotations/instances_val2017.json')
    
    if args.countall:
        list_all_classes_with_size(train_coco, val_coco)
        exit(0)
    
    if args.countonly:
        if not classes:
            print("Error: Please specify classes when using --countonly")
            exit(1)
        for className in classes:
            train_count, train_size = count_class_images_with_size(className, train_coco, "train")
            val_count, val_size = count_class_images_with_size(className, val_coco, "val")
            total_count = train_count + val_count
            total_size = format_size(train_size + val_size)
            print(f"{className}: {train_count} train, {val_count} val, {total_count} total, ~{total_size}")
        exit(0)

# Validate classes for download
if not classes:
    print("Error: Please specify classes to download")
    exit(1)

if(classes[0] == "--help"):
    with open('YOLO-Coco-Dataset-Custom-Classes-Extractor/classes.txt', 'r') as fp:
        lines = fp.readlines()
    print("**** Classes ****\n")
    [print(x.split('\n')[0]) for x in lines]
    exit(0)     

print("Classes to download: ", classes)

# Load both train and val annotations
train_coco = COCO('annotations/instances_train2017.json')
val_coco = COCO('annotations/instances_val2017.json')

# Validate classes against train set
cats = train_coco.loadCats(train_coco.getCatIds())
nms=[cat['name'] for cat in cats]

for name in classes:
    if(name not in nms):
        print(f"{name} is not a valid class, Skipping.")
        classes.remove(name)

# Create main directories for each class
for className in classes:
    makeDirectory(f'{className}_dataset')
    makeDirectory(f'{className}_dataset/train')
    makeDirectory(f'{className}_dataset/valid')

threads = []

# Process train set
for className in classes:
    t = threading.Thread(target=getImagesFromClassName, args=(className, train_coco, "train", num_workers)) 
    threads.append(t)

# Process val set  
for className in classes:
    t = threading.Thread(target=getImagesFromClassName, args=(className, val_coco, "val", num_workers)) 
    threads.append(t)
    
for t in threads:
    t.start()

for t in threads:
    t.join()

# Create data.yaml files
for className in classes:
    create_data_yaml(className)

print("Done.")