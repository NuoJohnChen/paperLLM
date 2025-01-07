
import os
import sys
import json
from tqdm import tqdm
import shutil
from pathlib import Path
import argparse
sys.path.append('/shared/hdd/junyi/paperscore')

from PaperScore import evaluate_and_suggest_improvement
from PaperScore import ConcurrentTaskManager

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='get qa pair')
    parser.add_argument('--base_dir', type=str, default='iclr24_8shot',
                      help='dir')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                      help='gpt-4o-mini or Qwen-72b-Instruct')
    args = parser.parse_args()
    
    # find all files in `sample_papers`
    sections_criteria = [
        './rules/rule-top-priority-core_rules_by_sections/abstract.json',
        './rules/rule-top-priority-core_rules_by_sections/background.json',
        './rules/rule-top-priority-core_rules_by_sections/conclusion.json',
        './rules/rule-top-priority-core_rules_by_sections/evaluation.json',
        './rules/rule-top-priority-core_rules_by_sections/introduction.json',
        './rules/rule-top-priority-core_rules_by_sections/title.json',
    ]
    BASE_DIR= args.base_dir
    #BASE_DIR = "/shared/hdd/junyi/prof_paper_zining"#'/shared/hdd/junyi/prof_papers_test_o1'#'/shared/hdd/data/openreview_data/ICLR.cc/2024/Conference'#'/shared/hdd/junyi/prof_papers_test'
    file_list = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith("paper.mmd"):
                file_list.append(os.path.join(root, file))

    for file in tqdm(file_list, desc="Revising papers"):
        with open(file, "r") as f:
            paper_whole_content = f.read()
        file_directory = os.path.dirname(file)
        file_directory = os.path.join(file_directory, "paper_revision")
        print(f"Processing file : {file_directory}")
        with ConcurrentTaskManager(max_workers=4) as manager:
            for section_criteria in sections_criteria:
                section_name = section_criteria.split("/")[-1].replace(".json", "")
                file_section_directory = os.path.join(file_directory, section_name)
                os.makedirs(file_section_directory, exist_ok=True)
                print(f"Processing section: {section_criteria}")
                with open(section_criteria, "r") as f:
                    criteria_array = json.load(f)

                    for criteria in criteria_array:
                        pass
                        # print(f"Submit evaluating and revising task for criteria: {criteria['category']}")
                        # evaluate_and_suggest_revision(client=None, paper_content=paper_whole_content, paper_type="academic research paper", criteria=criteria, output_directory=file_section_directory)

                        # manager.submit_task(evaluate_and_suggest_improvement, client=None, paper_content=paper_whole_content, paper_type="academic research paper", criteria=criteria, output_directory=file_section_directory, model='/disk1/nuochen/models/Qwen2.5-7B-Instruct')
                        manager.submit_task(evaluate_and_suggest_improvement, client=None, paper_content=paper_whole_content, paper_type="academic research paper", criteria=criteria, section=section_name,output_directory=file_section_directory, model=args.model)#'o1-mini-2024-09-12')
                    print("Waiting for all tasks to complete")
                    results = manager.get_results()
if __name__ == "__main__":
    main()
