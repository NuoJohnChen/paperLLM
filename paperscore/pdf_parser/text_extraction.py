import PyPDF2
import pymupdf
import re
import json
from loguru import logger
import openai
from src.agent.base import Agent

SECTION_MAPPING = "pdf_parser/section_mapping.json"
list_of_major_sections = [
    "Background",
    "Related Work",
    "Motivation",
    "Technical Approach / Methodology",
    "Evaluation / Results",
    "Discussion",
    "Conclusion",
    "Future Work",
    "References",
    "Applications",
]


def extract_title_from_metadata(pdf_path):
    title = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        # Get the document's metadata
        metadata = reader.metadata
        if metadata and "/Title" in metadata:
            if metadata["/Title"] != "":
                title = metadata["/Title"]
                return title
        else:
            logger.warning("No title found in metadata.")
        try:
            doc = pymupdf.open(pdf_path)
            first_page = doc[0]
            text = first_page.get_text()
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return " ".join(lines[:2])
        except Exception as e:
            logger.warning(f"Exception encountered: {e}")
            return title


def create_full_text(json_data):
    # Extract the abstract text
    abstract_text = json_data.get("abstractText", {}).get("text", "")

    # Initialize full text with abstract
    full_text = abstract_text + "\n\n"

    # Iterate through sections
    for section in json_data.get("sections", []):
        # Add section title if present
        if isinstance(section, dict) and "title" in section and isinstance(section["title"], dict):
            full_text += section["title"].get("text", "") + "\n\n"

        # Add paragraphs
        if isinstance(section, dict) and "paragraphs" in section:
            for paragraph in section["paragraphs"]:
                if isinstance(paragraph, dict) and "text" in paragraph:
                    full_text += paragraph["text"] + " "
            full_text += "\n\n"

    return full_text.strip()


def process_json(json_data):
    try:
        full_text = create_full_text(json_data)
        return full_text
    except Exception as e:
        print(f"Error processing JSON: {e}")
        return None


def get_section_positions(sections: dict) -> dict:
    """
    Creates a dictionary mapping section names to their position/order in the document.

    Args:
        sections (dict): Dictionary containing section names and their content

    Returns:
        dict: Dictionary mapping section names to their numerical position (1-based)
    """
    section_positions = {}
    for position, section_name in enumerate(sections.keys(), start=1):
        section_positions[section_name] = position
    return section_positions


def extract_sections(text, section_mapping_path=SECTION_MAPPING):

    style = detect_numbering_style(text)
    if style == "roman":
        structured_data = extract_sections_roman_numerals(text, section_mapping_path)
    else:
        structured_data = extract_sections_v1(text, section_mapping_path)
    if (len(structured_data) < 5) or (len(structured_data) > 12):
        structured_data = extract_sections_with_llm(text, section_mapping_path)
    section_positions = get_section_positions(structured_data)
    if structured_data.get("references", ""):
        structured_data = update_structured_data_with_appendices(structured_data)
        if structured_data.get("appendices", ""):
            section_positions["appendices"] = len(structured_data)
    return structured_data, section_positions


def detect_numbering_style(text):
    """
    Detects the numbering style used in the document sections.
    Returns: 'roman', 'numeric', or 'unnumbered'
    """
    # Common section patterns
    roman_pattern = r"((?:IX|IV|V?I{1,3}|V|X)\.)\s+[A-Z][A-Za-z\s\-&/]+"
    numeric_pattern = r"(\d+\.)\s+[A-Z][A-Za-z\s\-&/]+"

    # Count matches for each style
    roman_matches = len(re.findall(roman_pattern, text, re.MULTILINE))
    numeric_matches = len(re.findall(numeric_pattern, text, re.MULTILINE))

    # Decision logic
    if roman_matches > numeric_matches and roman_matches >= 2:
        return "roman"
    elif numeric_matches > roman_matches and numeric_matches >= 2:
        return "numeric"
    else:
        return "unnumbered"


def mapping_sections_with_llm(structured_data: dict, section_mapping_path: str = SECTION_MAPPING):
    with open(section_mapping_path, "r") as f:
        section_mappings = json.load(f)
    fixed_sections = ["title", "abstract", "introduction"]
    temp_structured_data = {}
    for k, v in structured_data.items():
        if k not in fixed_sections:
            temp_structured_data[k] = v
    prompt = f"""
        You are provided with two pieces of data:

        1. **structured_data**: This is a JSON object containing sections of a paper with their section headers and content.

        2. **section_mappings**: This is a JSON object that maps canonical section names to lists of possible variations of section titles.

        **Your Task:**

        - Analyze the `structured_data` and map each section title (i.e., the keys in `structured_data`) to the most appropriate canonical section name from `section_mappings`.
        - Produce a JSON object called `sections_mapping`, where the keys are the canonical section names, and the values are **lists of section titles** from `structured_data` that map to each canonical section.

        **Requirements:**

        - Ensure that each section title in `structured_data` (i.e., each key) is considered.
        - Use the variations in `section_mappings` to find the best match for each section title.
        - **Only consider the section titles provided as keys in `structured_data`. Do not extract or consider any subsections, headers, or titles from within the content of each section.**
        - **Do not extract any additional section titles or headings from the content.**
        - **The mapping can be one-to-many**, meaning multiple section titles from `structured_data` can map to the same canonical section.
        - **However, each section title from `structured_data` should be mapped to at most one canonical section. Do not map the same section title to multiple canonical sections.**
        - **Only use the canonical section names from `section_mappings` as keys in your output.**
        - If a section title does not match any canonical names or their variations, and you cannot determine a relevant mapping based on its content, **do not map it** (i.e., leave it unmapped).
        - **Do not map sections to canonical sections unless it is appropriate based on the title or content.**

        **Example:**

        If `structured_data` contains sections titled `"technical overview and examples"`, `"block-delayed sequences"`, and `"cost semantics"`, and they all relate to the methodology, you can map them all to the canonical section `"Technical Approach / Methodology"` like this:

        {{"Technical Approach / Methodology": ["technical overview and examples", "block-delayed sequences", "cost semantics"]}}

        **section_mappings:**
        {section_mappings}

        Instructions:

        - Read each section title in `structured_data` (i.e., the keys of the JSON object).
        - For each section, compare its title to the lists of variations under each canonical name in `section_mappings`.
        - Assign the section to the canonical section whose variations best match the section title.
        - If necessary, consider the content of the section to determine the best mapping.
        - **Do not extract or consider any subsections, headings, or titles from within the content of the sections in `structured_data`.**
        - **Each section title from `structured_data` should be assigned to at most one canonical section. Do not assign the same section title to multiple canonical sections.**
        - **If you cannot confidently map a section title to any canonical section name, do not include it in the `sections_mapping`.**
        - **Produce a JSON object `sections_mapping`, where the keys are the canonical section names from `section_mappings`, and the values are lists of section titles from `structured_data`.**

        **Output Format**:
        {{
            "sections_mapping": {{
                "Background": ["<section_title>"],
                "Related Work": ["<section_title>"],
                "Motivation": ["<section_title>", "<section_title>"],
                "Technical Approach / Methodology": ["<section_title>", "<section_title>", "<section_title>"],
                "Evaluation / Results": ["<section_title>"],
                "Conclusion": ["<section_title>"]
                // Only include mappings you are confident about
            }}
        }}

        Now, please perform the mapping and provide the `sections_mapping` JSON object.
        **Structured Data**
        {temp_structured_data}

        """
    try:
        agent = Agent("mapping_sections_with_llm", tags=["pdf_parser"])
        result = agent.run(
            prompt=[
                {
                    "role": "system",
                    "content": "You are an AI language model assistant that specializes in parsing and understanding academic papers.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format="json_object",
        )

        sections_mapping = json.loads(result)["sections_mapping"]
        for k, v in structured_data.items():
            if k not in [
                item
                for sublist in sections_mapping.values()
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]:
                sections_mapping[k] = [k]
        return sections_mapping
    except Exception as e:
        logger.error(f"Error mapping sections with LLM: {e}")
        return {}


def extract_sections_with_llm(text: str, section_mapping_path: str = SECTION_MAPPING):
    # Load section_mapping.json
    with open(section_mapping_path, "r") as f:
        section_mapping = json.load(f)
    sections = {}
    sections_mapping = {}
    # Build a flat list of possible headings from the section mapping
    possible_headings = []
    for headings_list in section_mapping.values():
        possible_headings.extend(headings_list)

    # Create a detailed prompt for the LLM
    # prompt = f"""
    #         Based on the provided full text of an academic paper between <BEGIN TEXT> and <END TEXT> tags, your task is to:

    #         1. Carefully read the text and identify all the section headers. Section headers may or may not be on their own lines and might not be separated by newlines. They could be numbered with Roman numerals (I, II, III, etc.), Arabic numerals (1, 2, 3, etc.), or unnumbered.

    #         2. For each section header, extract:
    #         - The exact section title as it appears in the text.
    #         - The starting character index (0-based) of the section content (the text immediately after the section header).
    #         - The ending character index (0-based) of the section content (up to but not including the next section header).
    #         - The section number if available.

    #         3. Return the extracted information as a JSON array of objects in the order the sections appear in the text. Each object should have the following fields:
    #         - "section_title": The exact title of the section.
    #         - "start_index": The starting character index of the section content.
    #         - "end_index": The ending character index of the section content.
    #         - "section_number": The section number as a string (e.g., "I", "1.2.3"), or null if unnumbered.

    #         4. Ensure that the indices correctly correspond to the positions in the text.

    #         5. Only include sections that are relevant to the main content of the paper (e.g., "Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion", "References", etc.). Use the following list of common section headings as a guide:

    #         {possible_headings}

    #         6. Do not include any sections from footnotes, endnotes, references, or acknowledgments unless they are main sections.

    #         7. Do not include any additional text, explanations, or commentary in your response.

    #         Here is the text:

    #         <BEGIN TEXT>
    #         {text}
    #         <END TEXT>
    #         """
    prompt = f"""
            Based on the provided full text of an academic paper between <BEGIN TEXT> and <END TEXT> tags, your task is to:

            1. Carefully read the text and identify all the **main section headers**. Section headers may or may not be on their own lines and might not be separated by newlines. They could be numbered with Roman numerals (I, II, III, etc.), Arabic numerals (1, 2, 3, etc.), or unnumbered.

            2. **Only consider the top-level (main) sections. Do not include subsections, subheadings, or any content under the main sections.**

            - **Main sections** are primary divisions of the paper, such as "Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion", etc.
            - **Subsections** are divisions within main sections, often numbered like "1.1", "II.A", "3.2.1", or labeled with letters like "A.", "B.", etc.
            - **Do not include** any subsections or subheadings in your output.

            3. For each **main section header**, extract:
            - The exact section title as it appears in the text.
            - The starting character index (0-based) of the section content (the text immediately after the main section header).
            - The ending character index (0-based) of the section content (up to but not including the next main section header).
            - The section number if available (e.g., "I", "2", "III"), or `null` if unnumbered.

            4. Return the extracted information as a JSON array of objects in the order the sections appear in the text. Each object should have the following fields:
            - `"section_title"`: The exact title of the **main section**.
            - `"start_index"`: The starting character index of the section content.
            - `"end_index"`: The ending character index of the section content.
            - `"section_number"`: The section number as a string, or `null` if unnumbered.

            5. Ensure that the indices correctly correspond to the positions in the text.

            6. Only include sections that are relevant to the main content of the paper (e.g., "Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion", "References", etc.). Use the following list of common section headings as a guide:

            {possible_headings}

            7. Do not include any sections from footnotes, endnotes, acknowledgments, or appendices unless they are main sections.

            8. **Example**:

            Given the following text snippet:

            I. INTRODUCTION
            This is the introduction section.

            II. METHODOLOGY
            This is the methodology section.

            A. Data Collection
            Details about data collection.

            III. RESULTS
            Results are presented here.
            Your output should be:

             {{
                "sections": [
                    {{
                        "section_title": "INTRODUCTION",
                        "start_index": 16,
                        "end_index": 48,
                        "section_number": "I"
                    }},
                    {{
                        "section_title": "METHODOLOGY",
                        "start_index": 65,
                        "end_index": 95,
                        "section_number": "II"
                    }},
                    {{
                        "section_title": "RESULTS",
                        "start_index": 140,
                        "end_index": 165,
                        "section_number": "III"
                    }}
                ]
            }}
            Note: Do not include the subsection “A. Data Collection” in your output.

	        9.	Do not include any additional text, explanations, or commentary in your response.

            Here is the text:
            <BEGIN TEXT>
            {text}
            <END TEXT>

            """
    try:
        logger.info("Extracting section content with LLM")

        agent = Agent("extract_sections_with_llm", tags=["pdf_parser"])
        result = agent.run(
            prompt=[
                {
                    "role": "system",
                    "content": "You are an AI language model assistant that specializes in parsing and understanding academic papers.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format="json_object",
        )

        sections_info = json.loads(result)
        possible_headings = {}
        for canonical_name, headings_list in section_mapping.items():
            for heading in headings_list:
                possible_headings[heading.lower()] = canonical_name.lower()

        # Process sections sequentially
        section_list = sections_info["sections"]
        for i, section in enumerate(section_list):
            title = section["section_title"].strip()
            # Create pattern that matches the exact section title
            # Escape special characters and make it case insensitive
            pattern = re.escape(title)
            title_matches = list(re.finditer(pattern, text, re.IGNORECASE))

            if title_matches:
                # Get the start position after the section title
                content_start = title_matches[0].end()

                # Find the end position (either next section or end of text)
                if i < len(section_list) - 1:
                    next_title = section_list[i + 1]["section_title"].strip()
                    next_matches = list(re.finditer(re.escape(next_title), text, re.IGNORECASE))
                    if next_matches:
                        content_end = next_matches[0].start()
                    else:
                        content_end = len(text)
                else:
                    content_end = len(text)

                # Extract and clean the content
                content = text[content_start:content_end].strip()

                # Process the title
                clean_title = re.sub(r"^(?:[IVX]+\.|[0-9]+\.)\s*", "", title)
                normalized_title = clean_title.lower()

                # Map to canonical name
                # canonical_name = possible_headings.get(normalized_title, normalized_title)

                sections[normalized_title] = content
                # sections_mapping[canonical_name] = title

        return sections

    except Exception as e:
        logger.error(f"Error extracting sections with LLM: {e}")
        result = {}
    return result


def extract_sections_roman_numerals(text, section_mapping_path=SECTION_MAPPING):
    # Load section_mapping.json
    with open(section_mapping_path, "r") as f:
        section_mapping = json.load(f)

    # Initialize variables
    sections = {}
    sections_mapping = {}

    # Build possible_headings mapping
    possible_headings = {}
    for canonical_name, headings_list in section_mapping.items():
        for heading in headings_list:
            possible_headings[heading.lower()] = heading.lower()  # Normalize to lowercase

    # Extract abstract
    abstract_match = re.search(r"Abstract[\s—–:\-]*", text, re.IGNORECASE)
    if abstract_match:
        abstract_start = abstract_match.end()
        # Find "Introduction" (may not be at the beginning of a line)
        intro_match = re.search(r"(I\.\s+)?Introduction", text[abstract_start:], re.IGNORECASE)
        if intro_match:
            intro_start = abstract_start + intro_match.start()
            abstract_end = intro_start
        else:
            abstract_end = len(text)
        abstract_text = text[abstract_start:abstract_end].strip()
        sections["abstract"] = abstract_text
    else:
        sections["abstract"] = ""

    # Extract Introduction
    # Since "Introduction" may not be at the beginning of a line, search for it anywhere
    intro_match = re.search(r"(I\.\s+)?Introduction", text, re.IGNORECASE)
    if intro_match:
        intro_start = intro_match.end()
        # Find the next section heading that starts at the beginning of a line
        # Define the section heading pattern that requires headings to start at the beginning of a line
        next_section_heading_pattern = re.compile(
            r"^\s*(?P<num>([IVXLCDM]+\.)|(\d+\.?)+)\s+(?P<title>[A-Z][A-Za-z0-9\s\-&/]+)", re.MULTILINE
        )
        next_section_match = next_section_heading_pattern.search(text, pos=intro_start)
        if next_section_match:
            intro_end = next_section_match.start()
        else:
            intro_end = len(text)
        introduction_text = text[intro_start:intro_end].strip()
        sections["introduction"] = introduction_text
        content_start = intro_end

    # Define the section heading pattern
    # section_heading_pattern = re.compile(
    #     r"(?P<num>([IVX]+\.|\d+\.)\s+)?(?P<title>[A-Z][A-Z0-9\s\-&/]+)", re.IGNORECASE
    # )
    section_heading_pattern = re.compile(
        r"^\s*(?P<num>([IVX]+\.|\d+\.)\s+)?(?P<title>[A-Z][A-Z0-9\s\-&/]+)\s*$", re.MULTILINE
    )

    # Find all section headings from content_start onwards
    section_headings = []
    for match in section_heading_pattern.finditer(text, pos=content_start):
        heading = match.group("title").strip()
        heading_start = match.start()
        heading_end = match.end()
        section_headings.append({"heading": heading, "start": heading_start, "end": heading_end})

    # Extract content between section headings
    for i, heading_info in enumerate(section_headings):
        heading = heading_info["heading"]
        heading_start = heading_info["start"]
        heading_end = heading_info["end"]
        if i + 1 < len(section_headings):
            next_heading_start = section_headings[i + 1]["start"]
        else:
            next_heading_start = len(text)
        section_content = text[heading_end:next_heading_start].strip()
        normalized_heading = heading.lower()
        canonical_name = possible_headings.get(normalized_heading, normalized_heading)
        sections[normalized_heading] = section_content
    # for section, content in sections.items():
    #     normalized_heading = section.lower()
    #     canonical_name = possible_headings.get(normalized_heading, normalized_heading)
    #     sections_mapping[canonical_name] = normalized_heading

    return sections


def extract_sections_v1(text, section_mapping_path=SECTION_MAPPING):
    # Load section_mapping.json
    with open(section_mapping_path, "r") as f:
        section_mapping = json.load(f)

    # Initialize variables
    sections = {}
    sections_mapping = {}
    current_main_section = None
    current_content = []

    # Build possible_headings mapping
    possible_headings = {}
    for canonical_name, headings_list in section_mapping.items():
        for heading in headings_list:
            possible_headings[heading.lower()] = heading.lower()  # Normalize to lowercase

    # Split text into lines
    lines = text.split("\n")

    # Initialize title
    title_lines = []
    abstract_lines = []
    is_abstract = False
    is_intro = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check if the current line is an Abstract heading
        if re.match(r"(?i)(abstract)", line):
            is_abstract = True  # Start capturing lines for the abstract

        # Check if we have reached the introduction
        if re.match(r"(?i)(introduction|\d+\s+introduction)", line):
            is_intro = True  # Stop capturing lines for the abstract

        if is_abstract and not is_intro:
            abstract_lines.append(line)

        # When Introduction is encountered, stop and save
        if is_intro:
            break

        i += 1

    # Join the abstract lines and save it in sections
    sections["abstract"] = " ".join(abstract_lines).strip()

    # Continue processing the rest of the lines
    combined_lines = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue  # Skip empty lines

        combined_lines.append(line)
        i += 1

    # Define heading patterns
    main_heading_pattern = re.compile(r"^(\d+)\s+(.*)$")  # Matches headings like '1 INTRODUCTION'
    sub_heading_pattern = re.compile(r"^(\d+\.\d+)\s+(.*)$")  # Matches headings like '2.1 Preliminary'

    # Processing lines after abstract
    for idx, line in enumerate(combined_lines):
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Normalize line for matching
        normalized_line = line.lower()

        # Check for main numbered heading
        m_main = main_heading_pattern.match(line)
        if m_main and "." not in m_main.group(1):
            # Save current section
            if current_main_section is not None:
                content = " ".join(current_content).strip()
                if current_main_section in sections:
                    sections[current_main_section] += " " + content
                else:
                    sections[current_main_section] = content
            # Start new main section
            section_number = m_main.group(1)
            section_title = m_main.group(2).strip()
            section_title_lower = section_title.lower()
            current_main_section = possible_headings.get(section_title_lower, section_title_lower)
            current_content = []
            continue

        # Check for unnumbered heading (known headings)
        if normalized_line in possible_headings:
            # Save current section
            if current_main_section is not None:
                content = " ".join(current_content).strip()
                if current_main_section in sections:
                    sections[current_main_section] += " " + content
                else:
                    sections[current_main_section] = content
            # Start new main section
            current_main_section = possible_headings[normalized_line]
            current_content = []
            continue

        # Check for all uppercase headings (e.g., 'INTRODUCTION')
        if line.isupper():
            normalized_line = line.lower()
            # Save current section
            if current_main_section is not None:
                content = " ".join(current_content).strip()
                if current_main_section in sections:
                    sections[current_main_section] += " " + content
                else:
                    sections[current_main_section] = content
            # Start new main section
            current_main_section = possible_headings.get(normalized_line, normalized_line)
            current_content = []
            continue

        # For subheadings and other content
        m_sub = sub_heading_pattern.match(line)
        if m_sub:
            sub_section_number = m_sub.group(1)
            sub_section_title = m_sub.group(2).strip()
            current_content.append(f"{sub_section_number} {sub_section_title}")
            continue

        # Append line to current content
        if current_main_section is None:
            continue  # Skip content before any known heading
        current_content.append(line)

    # Save the last section
    if current_main_section is not None:
        content = " ".join(current_content).strip()
        if current_main_section in sections:
            sections[current_main_section] += " " + content
        else:
            sections[current_main_section] = content

    # for section, content in sections.items():
    #     normalized_heading = section.lower()
    #     canonical_name = possible_headings.get(normalized_heading, normalized_heading)
    #     sections_mapping[canonical_name] = normalized_heading

    return sections


def update_structured_data_with_appendices(structured_data: dict):
    reference_text = structured_data.get("references", "")

    prompt = f"""
    Given the following text from a research paper's references section, your task is to:
    1. Identify and separate actual references (citations starting with [n]) from any appendix content
    2. Return a JSON object containing both the cleaned references and any appendix content found

    Text to analyze:
    {reference_text}

    Rules for separation:
    - References must start with [n] where n is a number (e.g., [1], [2], etc.)
    - Include complete reference entries including their multi-line content
    - Any content that doesn't belong to numbered references should be considered appendix
    - Return empty string for appendices if none found

    Return ONLY a JSON object in this exact format:
    {{
        "references": "string containing only the numbered citations [1], [2], etc.",
        "appendices": "string containing any non-reference content found after references"
    }}
    """

    try:
        logger.info(f"Extracting appendices from references section")
        agent = Agent("extract_appendices", tags=["pdf_parser"])
        result = agent.run(
            prompt=[
                {
                    "role": "system",
                    "content": "You are an AI language model assistant that specializes in parsing and understanding academic papers.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format="json_object",
        )

        # Parse result and update structured_data
        parsed_result = json.loads(result) if isinstance(result, str) else result

        # Update references if found
        if parsed_result.get("references"):
            structured_data["references"] = parsed_result["references"].strip()

        # Add appendices if found and not empty
        if parsed_result.get("appendices", "").strip():
            structured_data["appendices"] = parsed_result["appendices"].strip()
        return structured_data

    except Exception as e:
        logger.error(f"Error processing LLM response in extract_appendices: {e}")
        return structured_data
