
import os
import sys
import json
from tqdm import tqdm

sys.path.append('/shared/hdd/junyi/paperscore')

from PaperScore import evaluate_and_suggest_revision
from PaperScore import ConcurrentTaskManager

def main():
    # find all files in `sample_papers`
    sections_criteria = [
        './rules/rule-top-priority-core_rules_by_sections/abstract.json',
        './rules/rule-top-priority-core_rules_by_sections/background.json',
        './rules/rule-top-priority-core_rules_by_sections/conclusion.json',
        './rules/rule-top-priority-core_rules_by_sections/evaluation.json',
        './rules/rule-top-priority-core_rules_by_sections/introduction.json',
        './rules/rule-top-priority-core_rules_by_sections/title.json',
    ]


    BASE_DIR = '/shared/hdd/junyi/prof_papers'
    file_list = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith("paper.mmd"):
                file_list.append(os.path.join(root, file))

    for file in tqdm(file_list, desc="Evaluating and revising papers"):
        with open(file, "r") as f:
            paper_whole_content = f.read()
        file_directory = os.path.dirname(file)
        file_directory = os.path.join(file_directory, "paper_scores")
        
        with ConcurrentTaskManager(max_workers=16) as manager:
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
                        manager.submit_task(evaluate_and_suggest_revision, client=None, paper_content=paper_whole_content, paper_type="academic research paper", criteria=criteria, output_directory=file_section_directory)

                    print("Waiting for all tasks to complete")
                    results = manager.get_results()
if __name__ == "__main__":
    main()
