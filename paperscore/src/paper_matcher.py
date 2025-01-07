from pdf_parser.process_pdf import process_pdf_task
from paper_search.paper_search import search_paper
from LatexParser import process_zip_file2, get_document_structure_latex, get_document_section_content_map, get_all_citations
import bibtexparser
from enum import Enum
from rich import print
from tqdm import tqdm
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class Venue(Enum):
    SIGMOD_RESEARCH_PAPERS_ALGORITHMS = ("SIGMOD", "SIGMOD Call for Research Papers (Algorithms)")
    SIGMOD_RESEARCH_PAPERS_SYSTEMS = ("SIGMOD", "SIGMOD Call for Research Papers (Systems)")
    SIGMOD_TUTORIAL_PROPOSALS = ("SIGMOD", "SIGMOD Call for Tutorial Proposals")
    SIGMOD_WORKSHOP_PROPOSALS = ("SIGMOD", "SIGMOD Call for Workshop Proposals")
    SIGMOD_DEMONSTRATION_PROPOSALS = ("SIGMOD", "SIGMOD Call for Demonstration Proposals")
    PODS_RESEARCH_PAPERS = ("PODS", "PODS Call for Research Papers")
    VLDB_RESEARCH_PAPERS = ("VLDB", "VLDB Call for Research Papers")
    ICML_RESEARCH_PAPERS = ("ICML", "ICML Call for Research Papers")

    def __init__(self, name, description):
        self._name_ = name
        self.description = description


def extract_section_content(*, zip_file_path: str):
    latex_original, bib_library = process_zip_file2(zip_file_path, "./temp")
    latex_cleaned = latex_original
    document_structure = get_document_structure_latex(latex_cleaned)

    section_content_map = get_document_section_content_map(document_structure)
    section_content_map = {k.lower(): v for k, v in section_content_map.items()}
    # print(document_structure.keys())
    # print(section_content_map.keys())
    TARGET_SECTIONS = ["experiments", "experiment", "evaluation", "baseline"]
    section_content = ""
    for section in section_content_map:
        for t in TARGET_SECTIONS:
            if t in section:
                section_content += section_content_map[section]
    return section_content, bib_library


def process_bib_entry(bib_entry, cites) -> list:
    if not isinstance(bib_entry, bibtexparser.model.Entry):
        raise TypeError("Expected bib to be of type bibtexparser.model.Entry")
    if bib_entry.key not in cites:
        return []

    title, author, year, journal, booktitle, publisher, url, doi = extract_bib_fields(bib_entry)
    if journal is None and booktitle is None and publisher is None:
        return []

    ccf_a = []
    if (journal is None) or ("arxiv" not in journal.lower()):
        concated = concatenate_fields(journal, booktitle, publisher)
        paper_class = search_paper(title=concated, conference_journal=concated, top_n=1)
        if not paper_class:
            return []

        paper_class = paper_class[0]
        if paper_class and paper_class['category'] == 'A':
            update_paper_class(paper_class, title, author, year, url, doi, concated)
            ccf_a.append(paper_class)
    return ccf_a


def extract_bib_fields(bib_entry: bibtexparser.model.Entry):
    title = bib_entry.fields_dict.get("title", None)
    title = title.value.strip() if title else None

    author = bib_entry.fields_dict.get("author", None)
    author = author.value if author else None

    year = bib_entry.fields_dict.get("year", None)
    year = year.value if year else None

    journal = bib_entry.fields_dict.get("journal", None)
    journal = journal.value if journal else None

    booktitle = bib_entry.fields_dict.get("booktitle", None)
    booktitle = booktitle.value if booktitle else None

    publisher = bib_entry.fields_dict.get("publisher", None)
    publisher = publisher.value if publisher else None

    url = bib_entry.fields_dict.get("url", None)
    url = url.value if url else None

    doi = bib_entry.fields_dict.get("doi", None)
    doi = doi.value if doi else None

    return title, author, year, journal, booktitle, publisher, url, doi


def concatenate_fields(journal: str | None, booktitle: str | None, publisher: str | None) -> str:
    concated = ""
    if journal:
        concated += journal
    if booktitle:
        concated += " " + booktitle
    if publisher:
        concated += " " + publisher
    return concated.strip()


def update_paper_class(paper_class, title, author, year, url, doi, concated):
    paper_class['title'] = title
    paper_class['author'] = author
    paper_class['year'] = year
    paper_class['url'] = url
    paper_class['doi'] = doi
    paper_class['real_venue'] = concated


def paper_matcher(*, zip_file_path: str, target_venue: Venue) -> list:
    """
    Matches papers from a given zip file to a target venue based on citation analysis.

    Args:
        zip_file_path (str): The file path to the zip file containing the papers.
        target_venue (Venue): The target venue to match papers against.

    Returns:
        None: This function does not return a value. It processes and filters papers based on the target venue.
    """
    content, biblibrary = extract_section_content(zip_file_path=zip_file_path)
    cites = get_all_citations(content)

    ccf_a = []
    for bib_entry in tqdm(biblibrary.entries):
        ccf_a.extend(process_bib_entry(bib_entry, cites))

    target_venue_papers = []
    for ccfs in ccf_a:
        if ccfs['row']['abbreviation'] == target_venue._name_:
            target_venue_papers.append(ccfs)
    return target_venue_papers


def get_example_sections(*, zip_file_path: str, target_venue: Venue, user_provided_reference_pdfs: list[str], user_previously_accepted_pdfs: list[str]) -> dict:
    final_result = {
        'zip_file_path': zip_file_path,
        'target_venue': str(target_venue),
        'matched_papers': [],
        'user_provided_reference_pdfs': {},
        'user_previously_accepted_pdfs': {},
    }

    matched_papers = paper_matcher(zip_file_path=zip_file_path, target_venue=target_venue)
    final_result['matched_papers'] = matched_papers

    def remove_unwanted_sections(sections: dict) -> dict:
        sections.pop('abstract', None)
        sections.pop('introduction', None)
        sections.pop('title', None)
        keys_to_remove = [section_name for section_name in sections
                          if 'background' in section_name
                          or 'relate' in section_name
                          or 'reference' in section_name
                          or 'conclusion' in section_name
                          ]
        for key in keys_to_remove:
            sections.pop(key, None)
        return sections.copy()

    # TODO: get example section from user_historical_accepted_pdf (Top1 priority)
    for pdf in user_previously_accepted_pdfs:
        result = process_pdf_task(pdf, output_path="./temp")
        sections = result['structured_data']
        sections = remove_unwanted_sections(sections)
        final_result['user_previously_accepted_pdfs'][pdf] = sections

    # TODO: get example section from user_provided_pdfs (Top2 priority)
    for pdf in user_provided_reference_pdfs:
        result = process_pdf_task(pdf, output_path="./temp")
        sections = result['structured_data']
        sections = remove_unwanted_sections(sections)
        final_result['user_previously_accepted_pdfs'][pdf] = sections

    # TODO: get example section from matched_papers (Top3 priority)

    return final_result


def main():
    zip_file_path = "/Users/junyi/Desktop/pd-core-latex/latex/fedsim.zip"
    get_example_sections(
        zip_file_path=zip_file_path,
        target_venue=Venue.SIGMOD_RESEARCH_PAPERS_SYSTEMS,
        user_provided_reference_pdfs=[],
        user_previously_accepted_pdfs=[]
    )


if __name__ == "__main__":
    main()

    # main("/Users/junyi/Desktop/pd-core-latex/latex/label-skew.zip")

    # main("/Users/junyi/Desktop/pd-core-latex/latex/llm-pbe.zip")
    # main("/Users/junyi/Desktop/pd-core-latex/latex/niid-bench.zip")

    # main("/Users/junyi/Desktop/pd-core-latex/latex/thundering.zip")
