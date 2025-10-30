#!/usr/bin/env python3
"""
Dataset Splitter for YOLO format datasets
Split large datasets into smaller chunks with anti-forgetting overlap
"""

import argparse
import os
import sys
import math
import random
import shutil
from pathlib import Path

def calculate_overlap_count(overlap_value, chunk_size):
    """Return absolute count from int/float input"""
    if overlap_value >= 1:
        return int(overlap_value)
    else:
        return int(overlap_value * chunk_size)

def estimate_total_chunks(total_images, chunk_size, overlap_count, decay):
    """Estimate total chunks needed with overlap + decay"""
    if total_images <= chunk_size:
        return 1
    
    remaining = total_images - chunk_size  # First chunk
    chunks = 1
    
    while remaining > 0:
        # Calculate total overlap for this chunk
        total_overlap = 0
        for i in range(chunks):
            overlap_from_chunk = overlap_count * (decay ** (chunks - i))
            total_overlap += overlap_from_chunk
        
        new_images_capacity = max(100, chunk_size - total_overlap)  # Minimum 100 new images
        remaining -= new_images_capacity
        chunks += 1
        
        if chunks > 1000:  # Safety break
            break
    
    return chunks

def check_efficiency_warning(total_images, chunk_size, estimated_chunks):
    """Warn if ratio > 1.5, ask user confirmation"""
    base_chunks = math.ceil(total_images / chunk_size)
    ratio = estimated_chunks / base_chunks
    
    if ratio > 1.5:
        print(f"⚠️  WARNING: Overlap settings require {estimated_chunks} chunks vs {base_chunks} base chunks (ratio: {ratio:.2f})")
        print("   This may be inefficient. Consider reducing --overlap or --decay.")
        
        response = input("Continue anyway? (Y/n): ").strip().lower()
        if response == 'n' or response == 'no':
            print("Aborted. Try with smaller --overlap or --decay values.")
            sys.exit(1)
        elif response == '' or response == 'y' or response == 'yes':
            print("Continuing with current settings...")
        else:
            print("Invalid input. Assuming 'no' and aborting.")
            sys.exit(1)

def get_overlap_samples(previous_chunks_data, overlap_count, decay, chunk_idx):
    """Return overlap samples from all previous chunks with decay"""
    overlap_files = []
    
    for i in range(chunk_idx):
        previous_chunk_idx = chunk_idx - 1 - i
        overlap_from_chunk = int(overlap_count * (decay ** (i + 1)))
        
        if overlap_from_chunk > 0 and previous_chunk_idx < len(previous_chunks_data):
            available_files = previous_chunks_data[previous_chunk_idx]
            sample_count = min(overlap_from_chunk, len(available_files))
            if sample_count > 0:
                samples = random.sample(available_files, sample_count)
                overlap_files.extend(samples)
    
    return overlap_files

def create_chunk(chunk_idx, input_dir, output_prefix, train_files, valid_files, train_overlap, valid_overlap, args):
    """Create chunk directory, copy/link files, create data.yaml with metadata"""
    chunk_name = f"{output_prefix}_chunk{chunk_idx}"
    
    # Create directory structure
    chunk_path = Path(chunk_name)
    (chunk_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (chunk_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (chunk_path / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (chunk_path / "valid" / "labels").mkdir(parents=True, exist_ok=True)
    
    # Combine base files with overlap
    all_train_files = train_files + train_overlap
    all_valid_files = valid_files + valid_overlap
    
    # Copy/link train files
    for img_file in all_train_files:
        img_name = os.path.basename(img_file)
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        
        # Source paths
        src_img = os.path.join(input_dir, "train", "images", img_name)
        src_label = os.path.join(input_dir, "train", "labels", label_name)
        
        # Destination paths
        dst_img = chunk_path / "train" / "images" / img_name
        dst_label = chunk_path / "train" / "labels" / label_name
        
        # Copy or symlink
        if args.symlink:
            if os.path.exists(src_img):
                os.symlink(os.path.abspath(src_img), dst_img)
            if os.path.exists(src_label):
                os.symlink(os.path.abspath(src_label), dst_label)
        else:
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    # Copy/link valid files
    for img_file in all_valid_files:
        img_name = os.path.basename(img_file)
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        
        # Source paths
        src_img = os.path.join(input_dir, "valid", "images", img_name)
        src_label = os.path.join(input_dir, "valid", "labels", label_name)
        
        # Destination paths
        dst_img = chunk_path / "valid" / "images" / img_name
        dst_label = chunk_path / "valid" / "labels" / label_name
        
        # Copy or symlink
        if args.symlink:
            if os.path.exists(src_img):
                os.symlink(os.path.abspath(src_img), dst_img)
            if os.path.exists(src_label):
                os.symlink(os.path.abspath(src_label), dst_label)
        else:
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    # Read original data.yaml for class info
    original_yaml_path = os.path.join(input_dir, "data.yaml")
    nc = 1
    names = ['class']
    
    if os.path.exists(original_yaml_path):
        with open(original_yaml_path, 'r') as f:
            for line in f:
                if line.strip().startswith('nc:'):
                    nc = int(line.split(':')[1].strip())
                elif line.strip().startswith('names:'):
                    names_str = line.split(':', 1)[1].strip()
                    try:
                        names = eval(names_str)
                    except:
                        names = ['class']
    
    # Create data.yaml with metadata
    yaml_content = f"""path: {chunk_name}
train: train/images
val: valid/images

nc: {nc}
names: {names}

# Chunk metadata
chunk_index: {chunk_idx}
chunk_size: {args.chunk_size}
overlap: {args.overlap}
decay: {args.decay}
source_dataset: {os.path.abspath(input_dir)}
"""
    
    with open(chunk_path / "data.yaml", 'w') as f:
        f.write(yaml_content)

def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset into chunks with anti-forgetting overlap")
    parser.add_argument("input", help="Input YOLO dataset directory")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Target size for each chunk (default: 3000)")
    parser.add_argument("--overlap", type=float, default=0.15, help="Overlap size: if >=1 then count, if <1 then fraction of chunk size")
    parser.add_argument("--decay", type=float, default=0.5, help="Decay factor for older chunks")
    parser.add_argument("--output", help="Output directory prefix (default: input dir name)")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying files")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        sys.exit(1)
    
    # Check for required subdirectories
    train_img_dir = os.path.join(args.input, "train", "images")
    valid_img_dir = os.path.join(args.input, "valid", "images")
    
    if not os.path.exists(train_img_dir):
        print(f"Error: {train_img_dir} not found")
        sys.exit(1)
    
    if not os.path.exists(valid_img_dir):
        print(f"Error: {valid_img_dir} not found")
        sys.exit(1)
    
    # Get image lists
    train_images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    valid_images = [f for f in os.listdir(valid_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    total_train = len(train_images)
    total_valid = len(valid_images)
    
    print(f"Total train images: {total_train}, valid images: {total_valid}")
    
    if total_train == 0:
        print("Error: No images found in train directory")
        sys.exit(1)
    
    # Calculate overlap count
    overlap_count = calculate_overlap_count(args.overlap, args.chunk_size)
    
    # Estimate total chunks and check efficiency
    estimated_chunks = estimate_total_chunks(total_train, args.chunk_size, overlap_count, args.decay)
    print(f"Estimated {estimated_chunks} chunks needed")
    
    check_efficiency_warning(total_train, args.chunk_size, estimated_chunks)
    
    # Set output prefix
    output_prefix = args.output if args.output else os.path.basename(args.input.rstrip('/'))
    
    # Create chunks with overlap using clean logic
    chunk_idx = 0
    train_images_used = 0
    valid_images_used = 0
    all_chunk_sizes = []  # Track actual chunk sizes for overlap calculation
    
    # If validation images < 100, reuse same validation set for all chunks
    reuse_validation = total_valid < 100
    if reuse_validation:
        print(f"Validation images ({total_valid}) < 100, reusing same validation set for all chunks")
    
    while train_images_used < total_train:
        # Calculate overlap from ALL previous chunks
        total_train_overlap = 0
        total_valid_overlap = 0
        train_overlap_files = []
        valid_overlap_files = []
        
        for prev_idx in range(chunk_idx):
            prev_chunk_size = all_chunk_sizes[prev_idx]
            
            # Train overlap
            train_overlap_from_prev = int(prev_chunk_size * args.overlap * (args.decay ** max(0, chunk_idx - prev_idx - 1)))
            if train_overlap_from_prev > 0:
                # Get random samples from previous chunk's range
                prev_start = sum(all_chunk_sizes[i] for i in range(prev_idx)) - sum(
                    int(all_chunk_sizes[j] * args.overlap * sum(args.decay ** (j - k) for k in range(j))) 
                    for j in range(prev_idx)
                )
                prev_end = prev_start + prev_chunk_size
                prev_train_files = train_images[max(0, prev_start):min(len(train_images), prev_end)]
                
                if len(prev_train_files) > 0:
                    sample_count = min(train_overlap_from_prev, len(prev_train_files))
                    overlap_samples = random.sample(prev_train_files, sample_count)
                    train_overlap_files.extend(overlap_samples)
                    total_train_overlap += len(overlap_samples)
            
            # Valid overlap (only if not reusing validation)
            if total_valid > 0 and not reuse_validation:
                valid_overlap_from_prev = int(train_overlap_from_prev * (total_valid / total_train))
                if valid_overlap_from_prev > 0:
                    valid_start = int(prev_idx * total_valid / estimated_chunks)
                    valid_end = int((prev_idx + 1) * total_valid / estimated_chunks)
                    prev_valid_files = valid_images[valid_start:valid_end]
                    
                    if len(prev_valid_files) > 0:
                        sample_count = min(valid_overlap_from_prev, len(prev_valid_files))
                        overlap_samples = random.sample(prev_valid_files, sample_count)
                        valid_overlap_files.extend(overlap_samples)
                        total_valid_overlap += len(overlap_samples)
        
        # Calculate new images for this chunk
        new_train_count = min(args.chunk_size - total_train_overlap, total_train - train_images_used)
        
        if reuse_validation:
            # Reuse all validation images for every chunk
            new_valid_count = total_valid
            new_valid_files = valid_images
        else:
            # Split validation images normally
            new_valid_count = int(new_train_count * (total_valid / total_train)) if total_train > 0 else 0
            new_valid_count = min(new_valid_count, total_valid - valid_images_used)
            new_valid_files = valid_images[valid_images_used:valid_images_used + new_valid_count]
        
        # Get new train files
        new_train_files = train_images[train_images_used:train_images_used + new_train_count]
        
        # Total chunk size
        total_chunk_size = new_train_count + total_train_overlap
        all_chunk_sizes.append(total_chunk_size)
        
        # Progress output
        action = "linking" if args.symlink else "copying"
        if total_train_overlap > 0 or total_valid_overlap > 0:
            print(f"Creating chunk {chunk_idx}: {action} {new_train_count} train + {total_train_overlap} overlap, {new_valid_count} valid + {total_valid_overlap} overlap")
        else:
            print(f"Creating chunk {chunk_idx}: {action} {new_train_count} train, {new_valid_count} valid")
        
        # Create the chunk
        create_chunk(chunk_idx, args.input, output_prefix, new_train_files, new_valid_files, train_overlap_files, valid_overlap_files, args)
        
        # Update counters
        train_images_used += new_train_count
        if not reuse_validation:
            valid_images_used += new_valid_count
        chunk_idx += 1
        
        # Safety break
        if new_train_count <= 0 or chunk_idx > 1000:
            break
    
    print(f"Completed {chunk_idx} chunks")

if __name__ == "__main__":
    main()