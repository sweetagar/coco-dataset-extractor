# COCO Dataset Extractor

Download specific classes from the COCO dataset and convert them to YOLO format with ease.

## Features

- 🚀 **400% faster downloads** with configurable parallel workers
- 🔄 **Safe resume** - automatically handles interrupted downloads
- 📊 **Progress tracking** with current/total counts
- 📁 **YOLOv5-ready format** - train/valid directory structure
- 🐴 **Bug-fixed annotations** - correctly handles multiple objects per image
- 📈 **Quick counting** - estimate dataset sizes before downloading
- 🎯 **Class validation** - checks valid COCO classes before download

## Quick Start

### 1. Download COCO Annotations

First, download the COCO annotation files (~241MB):

```bash
# Download COCO 2017 annotations
# Option 1: Using wget
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Option 2: Using curl
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract annotations
unzip annotations_trainval2017.zip
```

This creates an `annotations/` folder with:
- `instances_train2017.json` (training set annotations)
- `instances_val2017.json` (validation set annotations)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download a Dataset

```bash
# Download horse dataset with 4 workers (default)
python coco-extractor.py horse

# Download with 8 parallel workers for faster speed
python coco-extractor.py horse --worker 8

# Download multiple classes
python coco-extractor.py horse dog cat --worker 6
```


### 4. Count Images Before Downloading

```bash
# Count specific classes
python coco-extractor.py horse dog --countonly
# Output: horse: 2941 train, 150 val, 3091 total, ~303.2MB

# List all available classes
python coco-extractor.py --countall
# Output: Lists all 80 COCO classes with counts and sizes
```


## Usage

### Basic Download
```bash
python coco-extractor.py <class_name> [options]
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--worker N` | Number of parallel download workers (default: 4) | `--worker 8` |
| `--countonly` | Count images and estimate size for specified classes (no download) | `horse dog --countonly` |
| `--countall` | List all 80 COCO classes with counts and sizes | `--countall` |

### Examples

#### Download Classes
```bash
# Single class with default 4 workers
python coco-extractor.py horse

# Single class with 8 workers (faster)
python coco-extractor.py horse --worker 8

# Multiple classes
python coco-extractor.py horse dog cat bicycle --worker 6
```

## Output Structure

The extractor creates a YOLOv5-ready dataset structure:

```
horse_dataset/
├── train/
│   ├── images/          # Training images
│   │   ├── 000000000009.jpg
│   │   └── 000000000042.jpg
│   └── labels/          # Training labels (YOLO format)
│       ├── 000000000009.txt
│       └── 000000000042.txt
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
└── data.yaml           # YOLOv5 configuration file
```

## Requirements

- Python 3.7+
- pycocotools
- requests  
- tqdm
- Internet connection for downloading images

## License

This tool is for educational and research purposes. Please respect COCO dataset license terms.

## Contributing

Feel free to submit issues and enhancement requests!