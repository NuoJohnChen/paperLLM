import os
import random
import shutil
from pathlib import Path
import argparse

def sample_folders(source_dir: str, target_dir: str, few_shot: int, seed: int = 42):
    """
    Randomly sample and copy specified number of folders from source directory to target directory
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        few_shot: Number of folders to sample
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all folders (excluding 'damaged' folder)
    folders = [f for f in os.listdir(source_dir) 
              if os.path.isdir(os.path.join(source_dir, f)) 
              and f != 'damaged']
    
    # Ensure sampling size doesn't exceed available folders
    few_shot = min(few_shot, len(folders))
    
    # Randomly sample specified number of folders
    selected_folders = random.sample(folders, few_shot)
    
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy selected folders to target directory
    for folder in selected_folders:
        src_path = os.path.join(source_dir, folder)
        dst_path = os.path.join(target_dir, folder)
        print(f"Copying folder: {folder}")
        shutil.copytree(src_path, dst_path)
    
    print(f"\nComplete! Copied {few_shot} folders to {target_dir}")
    return selected_folders

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Random folder sampling')
    parser.add_argument('--few_shot', type=int, default=8,
                      help='Number of folders to sample')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Setup paths
    source_dir = "/shared/hdd/data/openreview_data/ICLR.cc/2024/Conference"
    target_dir = f"/shared/hdd/junyi/iclr24_{args.few_shot}shot"
    
    # Execute sampling and copying
    selected = sample_folders(source_dir, target_dir, args.few_shot, args.seed)
    
    # Print results
    print("\nSelected folders:")
    for folder in selected:
        print(folder)

if __name__ == "__main__":
    main()