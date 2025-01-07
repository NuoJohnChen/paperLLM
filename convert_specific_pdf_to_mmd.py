import os
from tqdm import tqdm
import subprocess
BASE_DIR = '/shared/hdd/data/openreview_data/ICLR.cc/'

def process_pdf_files(pdf_files_part_file: str):
        
    with open(pdf_files_part_file, "r") as f:
        print(pdf_files_part_file)
        pdf_file = f.readlines()

    for pdfpath in tqdm(pdf_file):
        pdfpath = pdfpath.strip()
        if os.path.exists(pdfpath.replace("paper.pdf", "promptensemble.mmd")):
            print(pdfpath, "exists")
            continue
        
        pdfdir = os.path.dirname(pdfpath)
        print(f"nougat {pdfpath} -o {pdfdir}")
        
        result = subprocess.run(["nougat", pdfpath, "-o", pdfdir], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(result.stderr)
        # Check if the subprocess failed
        if result.returncode != 0:
            print(f"Failed to process {pdfpath} with nougat")


# python pre-process.py
# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python convert.py --pdf_files pdf_files_part_2.txt
# CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python convert.py --pdf_files pdf_files_part_1.txt

def main():
    # add argparse to convert.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_files", type=str, required=True) # default="pdf_files_part_1.txt"
    args = parser.parse_args()
    process_pdf_files(args.pdf_files)

if __name__ == "__main__":
    main()
