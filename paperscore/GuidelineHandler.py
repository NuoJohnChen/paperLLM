import re
import os
import difflib


def parse_guidelines(filename):
    """Parse guidelines from the text file."""
    allowed_sections = ['TITLE', 'ABSTRACT', 'INTRODUCTION', 'BACKGROUND', 'RELATED WORK', 'CONCLUSION', 'EVALUATION', 'DISCUSSION', 'FUTURE WORK', 'ACKNOWLEDGMENTS', 'REFERENCES', 'ENTIRE_PAPER', 'TECHNICAL_SECTION']
    guidelines = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    guideline_blocks = content.split('\n\n')
    for block in guideline_blocks:   
        guideline_content = block.strip()     
        if not guideline_content:  # Skip empty blocks
            continue
        target_section = extract_field(guideline_content, 'Target Section')
        if not target_section:
            target_section = extract_field(guideline_content, 'Category')
        
        # Find the closest match in allowed_sections
        if target_section.upper() not in allowed_sections:
            closest_match = difflib.get_close_matches(target_section.upper(), allowed_sections, n=1, cutoff=0.9)
            print(f"\nparse_guidelines, guideline: {guideline_content}, Closest match for {target_section}: {closest_match}\n")
            if closest_match:
                target_section = closest_match[0]
            else:
                print(f"\nparse_guidelines, guideline: {guideline_content}, No close match is found for {target_section}\n")
                continue
        
        fields = {
            'Mistake Item UUID': extract_field(guideline_content, 'Mistake Item UUID'),
            'Mistake Name': extract_field(guideline_content, 'Mistake Name'),
            'Target Section': target_section.upper(),
            'Description': extract_field(guideline_content, 'Description'),
            'How to Check': extract_field(guideline_content, 'How to Check'),
            'How to Solve': extract_field(guideline_content, 'How to Solve'),
        }
        guidelines.append(fields)
    return guidelines


def extract_field(text, field_name):
    """Extract a field from the guideline content."""
    pattern = r'\*\*' + re.escape(field_name) + r'\*\*:\s*(.*?)(?=\n\*\*|$)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ''


def guideline_to_string(guideline):
    """Convert a guideline dictionary to a formatted string."""
    template = """**Mistake Name**: {Mistake Name}
**Target Section**: {Target Section}
**Description**: {Description}
**How to Check**: {How to Check}
**How to Solve**: {How to Solve}"""
    
    return template.format(**guideline)


def get_guideline_string(guidelines, mistake_name):
    """Find a guideline by mistake name and return it as a string."""
    for guideline in guidelines:
        if guideline['Mistake Name'].lower() == mistake_name.lower():
            return guideline_to_string(guideline)
    return "Guideline not found."


def split_guidelines_into_sections(guidelines):
    """Split guidelines into sections based on the Target Section field."""
    sections = {}
    for guideline in guidelines:
        target_section = guideline['Target Section']
        sections[target_section] = sections.get(target_section, []) + [guideline]
    return sections

def split_guideline_file_into_sections(filename):
    #create an output directory based on the filename
    output_directory = filename.replace('.txt', '')+"_rules_by_sections"
    os.makedirs(output_directory, exist_ok=True)
    print(f"Guidelines have been split into sections and saved to the directory {output_directory}")
    guidelines = parse_guidelines(filename)
    guidelines_by_sections = split_guidelines_into_sections(guidelines)
    #store the guidelines of each section into a file, filename based on the input filename
    for section, guidelines_per_section in guidelines_by_sections.items():
        print(f"Section: {section}")
        section_filename = output_directory + '/' + section + '.txt'
        with open(section_filename, 'w', encoding='utf-8') as f:
            for guideline in guidelines_per_section:
                f.write(guideline_to_string(guideline) + '\n\n')
   
#test
def test_split_guidelines_into_sections():
    guidelines = parse_guidelines('rules/rule_background_relatedwork-refined.txt')
    guidelines_by_sections = split_guidelines_into_sections(guidelines)
    for section, guidelines_per_section in guidelines_by_sections.items():
        print(f"Section: {section}, guidelines_per_section: {guidelines_per_section}")
        for guideline in guidelines_per_section:
            print(guideline_to_string(guideline))


def test_split_guideline_file_into_sections():
    input_filename = 'rules/convert_ENTIRE_PAPER.txt'
    split_guideline_file_into_sections(input_filename)

#test_split_guideline_file_into_sections()
#print(guideline_to_string(guidelines[0]))
#test_split_guideline_file_into_sections()
