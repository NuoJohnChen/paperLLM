python 3.10
pip install langfuse
pip install python-dotenv
pip install openai
pip install commonmark
pip install loguru
pip install psycopg2-binary
pip install pymupdf

prompt is in paparscore/PaperScore.py

still include some personal paths. remove when publicing repo.

## Prepare Data
(1. convert_specific_pdf_to_mmd.py)
(2. extract_iclr24_fewshot.py)

## get_revise
python paperscore/revise.py

## show_html
python iclr24_8shot show_revise_to_html.py
