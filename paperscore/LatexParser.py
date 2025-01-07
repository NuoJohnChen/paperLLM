import zipfile
import os
import re
import json
import sys
import logging
import shutil  # Add this import
import traceback
import subprocess

logging.basicConfig(
    filename="LatexParser.log", level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
# clear the content of the log file first
if os.path.exists("LatexParser.log"):
    with open("LatexParser.log", "w") as f:
        f.truncate()


def clean_latex_content(latex_content, is_save_figtable=False):
    # Dictionary to store original figure and table content
    figtable_store = {}

    # Function to replace figure/table content with just the caption
    def replace_with_caption(match):
        full_match = match.group(0)
        environment = match.group(1)  # 'figure' or 'table'
        content = match.group(2)

        # Extract caption
        caption_match = re.search(r"\\caption\{(.*?)\}", content, re.DOTALL)

        if is_save_figtable:  # Store the full original fig & table content
            key = f"{environment}_{len(figtable_store)}"
            figtable_store[key] = full_match

            if caption_match:
                caption = caption_match.group(1)
                return f"\\begin{{{environment}}}\n\\caption{{{caption}}}\n\\CONTENT_PLACEHOLDER_{key}\n\\end{{{environment}}}"
            else:
                return f"\\begin{{{environment}}}\n\\CONTENT_PLACEHOLDER_{key}\n\\end{{{environment}}}"
        else:  # Simply remove the interior of fig & table
            if caption_match:
                caption = caption_match.group(1)
                return f"\\begin{{{environment}}}\n\\caption{{{caption}}}\n\\end{{{environment}}}"
            else:
                return f"\\begin{{{environment}}}\n\\end{{{environment}}}"

    # Replace figure and table content with captions (and placeholders)
    pattern = r"\\begin\{(figure|table)[\*]?\}(.*?)\\end\{\1[\*]?\}"
    cleaned_content = re.sub(pattern, replace_with_caption, latex_content, flags=re.DOTALL)
    return (cleaned_content, figtable_store) if is_save_figtable else cleaned_content


def restore_latex_content(cleaned_content, figtable_store):
    # Restore the original figure and table content
    for key, original_content in figtable_store.items():
        placeholder = f"\\CONTENT_PLACEHOLDER_{key}"
        cleaned_content = cleaned_content.replace(placeholder, original_content)

    return cleaned_content


# Step 1: Unzip the LaTeX text files
def unzip_files(zip_path, extract_to):
    # print(f"unzip_files: {zip_path} to {extract_to}")
    if os.path.exists(extract_to):
        print(f"Warning: Directory already exists: {extract_to}")
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


# Step 2: Identify the main LaTeX file that includes all other files
def find_main_latex_file(directory):
    main_file = None
    encodings = ["utf-8", "iso-8859-1", "windows-1252"]  # Add more encodings if needed

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                file_path = os.path.join(root, file)
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                            if r"\begin{document}" in content and r"\end{document}" in content:
                                main_file = file_path
                                logging.info(f"Main LaTeX file found: {main_file} (encoding: {encoding})")
                                return main_file
                    except UnicodeDecodeError:
                        continue  # Try the next encoding
                    except Exception as e:
                        logging.error(f"Error reading file {file_path}: {str(e)}")

    if not main_file:
        logging.warning("No main LaTeX file found.")
    return main_file


def find_bib_files(directory):
    bib_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".bib") and not file.startswith("._"):  # Ignore macOS 拉屎文件
                bib_files.append(os.path.join(root, file))
    return bib_files


# Step 3: Extract the order of included files from the main file
def extract_included_files(main_file):
    included_files = []
    with open(main_file, "r", encoding="utf-8") as f:
        content = f.read()
        # we need to remove comments from the content first
        content = remove_comments_and_empty_lines(content)
        # Look for \include{} or \input{} commands
        include_pattern = r"\\(?:include|input|subfile)\{(.*?)\}"
        included_files = re.findall(include_pattern, content)
    logging.info(f"Included files: {included_files}")
    return included_files


# Step 4: Combine LaTeX files according to the order in the main file
def combine_latex_files_in_order(included_files, directory):
    combined_text = ""
    print(f"combine_latex_files_in_order: {included_files}")
    print(f"directory: {directory}")

    for file in included_files:
        if not file.endswith(".tex"):
            file = file + ".tex"
        file_path = os.path.join(directory, f"{file}")  # Add .tex extension
        print(f"Processing file: {file_path}")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                filecontent = f.read()
                combined_text += filecontent + "\n\n"
                logging.info(f"File content: {filecontent}")
    logging.info(f"Combined text: {combined_text}")
    return combined_text


# Step 5: Remove LaTeX comments
def remove_empty_lines(text):
    # Split the text into lines, filter out empty lines, and join back
    return "\n".join(line for line in text.splitlines() if line.strip())


def remove_comments_and_empty_lines(latex_content):
    logging.info(f"remove_comments: {latex_content}")
    clear_content = re.sub(r"(?<!\\)%.*", "", latex_content)
    logging.info(f"clear_content: {clear_content}")
    clear_content = remove_empty_lines(clear_content)
    logging.info(f"clear_content after removing empty lines: {clear_content}")
    return clear_content


# Step 6: Identify the structure of the paper and split into sections
def get_document_structure_latex(latex_content):
    document_structure = {"title": "", "abstract": "", "sections": []}

    # Regular expressions to identify LaTeX components
    title_pattern = r"\\title\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    abstract_pattern = r"\\begin\{abstract\}(.*?)\\end\{abstract\}"
    section_pattern = r"\\section\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}(.*?)(?=\\section|\\end\{document\})"
    subsection_pattern = (
        r"\\subsection\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}(.*?)(?=\\subsection|\\section|\\end\{document\}|$)"
    )

    # Extract title and abstract (unchanged)
    title_match = re.search(title_pattern, latex_content, re.DOTALL)
    if title_match:
        document_structure["title"] = title_match.group(1).strip()

    abstract_match = re.search(abstract_pattern, latex_content, re.DOTALL)
    if abstract_match:
        document_structure["abstract"] = "\\begin{abstract}" + abstract_match.group(1).strip() + "\\end{abstract}"

    # Extract sections
    sections = re.findall(section_pattern, latex_content, re.DOTALL)
    for section_title, section_content in sections:
        section_data = {"title": section_title.strip(), "content": "", "subsections": []}

        # Extract subsections within this section
        subsections = re.findall(subsection_pattern, section_content, re.DOTALL)
        for subsection_title, subsection_content in subsections:
            subsection_data = {
                "title": subsection_title.strip(),
                "content": "\\subsection{" + subsection_title.strip() + "}" + subsection_content.strip(),
            }
            section_data["subsections"].append(subsection_data)

        # Remove subsection content from section content
        section_content_without_subsections = re.sub(subsection_pattern, "", section_content, flags=re.DOTALL)
        section_data["content"] = (
            "\\section{" + section_data["title"] + "}" + section_content_without_subsections.strip()
        )

        document_structure["sections"].append(section_data)

    return document_structure


# Step 7: Convert the structured content into JSON format
def convert_to_json(document_structure, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(document_structure, f, indent=4)
    logging.info(f"JSON structure saved to {output_file}")


def paste_text(filepath, extract_to, outfile):
    try:
        mergefiles(filepath, extract_to, outfile)
        # with open(filepath, "r", encoding='utf-8') as subfile:
        # file_content = subfile.read()
        # outfile.write(file_content)
        # logging.info(f"pasted file content: {file_content}")
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")


def mergefiles(main_file_name, extract_to, output_file_writer):
    with open(main_file_name, "r", encoding="utf-8") as infile:
        for line in infile:
            # find input lines that are not commented
            if "\\input" in line and (line.find("%") < 0 or line.find("%") > line.find("\\input")):
                m = re.search(r"\\input\{(.+?)\}", line)
                if m:
                    filepath = m.group(1)
                    # add tex extension if missing
                    if not filepath.endswith(".tex"):
                        filepath += ".tex"
                    # need to extend the output_file_name to the full path
                    filepath = os.path.join(extract_to, filepath)
                    logging.info(f"\\input pasting file: {filepath}")
                    paste_text(filepath, extract_to, output_file_writer)
            if "\\include" in line and (line.find("%") < 0 or line.find("%") > line.find("\\include")):
                m = re.search(r"\\include\{(.+?)\}", line)
                if m:
                    filepath = m.group(1)  # \include{FILE} does not require .tex extension in the FILE
                    filepath = os.path.join(extract_to, filepath + ".tex")
                    paste_text(filepath, extract_to, output_file_writer)
            else:
                output_file_writer.write(line)
        output_file_writer.write("\n")  # 防止最后一行没有换行符导致后面的内容与前面的注释在同一行


# Main function to execute the entire process
def process_zip_file(zip_path, output_file, is_clean=False):
    # need to give a different name to the extract_to directory each time according to the zip file name
    extract_to = f"unzipped_files_{os.path.splitext(os.path.basename(zip_path))[0]}"

    # Step 1: Unzip the file
    if is_clean:
        extract_to = os.path.join(output_file, extract_to)
    unzip_files(zip_path, extract_to)

    # Step 2: Find the main LaTeX file
    main_file = find_main_latex_file(extract_to)
    if not main_file:
        print("Main LaTeX file not found.")
        return
    # output the merged file to a file named after the zip file with a suffix of "_merged.tex"
    base_name = os.path.splitext(os.path.basename(zip_path))[0]
    output_file_merged = os.path.join(os.path.dirname(output_file), f"{base_name}_merged.tex")
    extract_to = "/".join(main_file.split("/")[:-1])

    with open(output_file_merged, "w", encoding="utf-8") as output_file_writer:
        mergefiles(main_file, extract_to, output_file_writer)

    # print the content of the merged file
    with open(output_file_merged, "r", encoding="utf-8") as f:
        logging.info(f.read())

    # read the combined_text from the merged file
    combined_text = ""
    with open(output_file_merged, "r", encoding="utf-8") as f:
        combined_text = f.read()

    # Step 5: Remove comments from the combined text
    cleaned_text = remove_comments_and_empty_lines(combined_text)
    # output this cleaned text to a file named after the zip file with a suffix of "_cleaned.tex"
    base_name = os.path.splitext(os.path.basename(zip_path))[0]
    output_file_cleaned = os.path.join(os.path.dirname(output_file), f"{base_name}_cleaned.tex")
    with open(output_file_cleaned, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # clear the extract_to directory
    if is_clean:
        clear_dir(extract_to)

    return cleaned_text


def process_zip_file2(zip_path, output_file) -> tuple[str, any]:
    extract_to = f"unzipped_files_{os.path.splitext(os.path.basename(zip_path))[0]}"
    latex_original = process_zip_file(zip_path, output_file)
    bib_files = find_bib_files(extract_to)

    bib_content = ""
    for bib_file in bib_files:
        print(f"bib_file: {bib_file}")
        with open(bib_file, "r", encoding="utf-8") as f:
            bib_content += f.read()

    import bibtexparser

    library = bibtexparser.parse_string(bib_content)
    return latex_original, library


def clear_dir(extract_to):
    if os.path.exists(extract_to):
        try:
            shutil.rmtree(extract_to, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not fully clear directory {extract_to}: {str(e)}")
            pass
        # for file in os.listdir(extract_to):
        #     os.remove(os.path.join(extract_to, file))


# New test function to process all zip files in a directory
def test_process_all_zip_files(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get all zip files in the input directory
    zip_files = [f for f in os.listdir(input_directory) if f.endswith(".zip")]

    if not zip_files:
        print(f"No zip files found in {input_directory}")
        return

    for zip_file in zip_files:
        zip_path = os.path.join(input_directory, zip_file)
        output_file = os.path.join(output_directory, f"{os.path.splitext(zip_file)[0]}_structured.json")

        print(f"Processing {zip_file}...")
        try:
            process_zip_file(zip_path, output_file)
            print(f"Successfully processed {zip_file}")
        except Exception as e:
            print(f"Error processing {zip_file}:")
            traceback.print_exc()  # This will print the full traceback

    print("All zip files processed.")


def get_full_section_content(section):
    content = section["content"]
    for subsection in section.get("subsections", []):
        content += "\n\n" + subsection["content"]
        # Recursively add content from nested subsections if they exist
        if "subsections" in subsection:
            content += "\n\n" + get_full_section_content(subsection)
    return content


def get_document_section_content_map(document_structure):
    """
    Creates a map of main section names (including title and abstract) to their content.

    :param document_structure: The parsed document structure
    :return: A dictionary mapping section names to their content
    """
    result = {"title": document_structure.get("title", ""), "abstract": document_structure.get("abstract", "")}

    if "sections" in document_structure:
        for section in document_structure["sections"]:
            result[section["title"]] = get_full_section_content(section)

    return result


# get all the titles of the sections in a document structure
def get_all_section_titles_latex(document_structure):
    titles = []
    # if the document structure has a title, add it to the titles
    if "title" in document_structure:
        titles.append("title")
    # if the document structure has a abstract, add it to the titles
    if "abstract" in document_structure:
        titles.append("abstract")
    if "sections" in document_structure:
        for section in document_structure["sections"]:
            titles.append(section["title"])
    return titles


def get_all_citations(latex_content: str) -> set[str]:
    citation_pattern = r"\\cite\{(.+?)\}"
    citations = re.findall(citation_pattern, latex_content)
    result_list = set()
    for citation in citations:
        result_list.update(citation.split(","))
    return result_list


# write a test function to test the get_document_structure_latex function
def test_get_document_structure_latex(zipfile: str):
    # Extract the base name of the zip file (without extension)
    base_name = os.path.splitext(os.path.basename(zipfile))[0]

    # Create a unique output directory based on the zip file name
    output_directory = f"rule_output/{base_name}"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    paper_latex_content_original = process_zip_file(zipfile, output_directory)
    document_structure = get_document_structure_latex(paper_latex_content_original)
    logging.info(document_structure)
    document_section_content_map = get_document_section_content_map(document_structure)
    for key, value in document_section_content_map.items():
        logging.info(f"document_section_content_map:{key}: {value}")
    all_section_titles = get_all_section_titles_latex(document_structure)
    logging.info(f"all_section_titles: {all_section_titles}")
    print(f"all_section_titles: {all_section_titles}")


def test_process_zip_file():
    input_directory = "latex_input"
    output_directory = "latex_output"
    test_process_all_zip_files(input_directory, output_directory)


if __name__ == "__main__":
    test_process_zip_file()
    # test_get_document_structure_latex(zipfile="latex/FeT NeurIPS 2024.zip")
    # test_get_document_structure_latex(zipfile="latex/test.zip")
    # test_get_document_structure_latex(zipfile="latex/AccelES-32cores.zip")
    # test_get_document_structure_latex(zipfile="latex/[ARR]Summarization Benchmark.zip")
    # test_get_document_structure_latex(zipfile="latex/Llamdex WWW 2025.zip")
    # test_get_document_structure_latex(zipfile="latex/WWW25-MSFGDA.zip")
