# PDF Parser

This project contains a PDF parsing tool that extracts text and metadata from PDF files.

## Prerequisites

- Python 3.10+
- Install dependencies through Poetry
- Java Runtime Environment (JRE) openjdk@11 or openjdk@17
- If local debug mode, add below into `.vscode/settings.json` (creat such file if don't have):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "justMyCode": true
        }
    ]
}
```
## Usage

To process a PDF file, use the `process_pdf_task` function in the `process_pdf.py` file. Here's how to use it:

1. Place your PDF file in the `pdfs` directory.
2. Set PYTHONPATH to the root directory of the project:
   ``` bash
   export PYTHONPATH=$(pwd)
   # OR
   export PYTHONPATH=/`pwd`
   ```
4. Call the function to use:
   ```
   from pdf_parser.process_pdf import process_pdf_task
   results = process_pdf_task('pdfs/Dupin__V1 (4).pdf', output_path = None)
   ```

   ### To generate rules for a pair of PDFs
   ```
   rules = process_pdf_and_generate_rules(
        client, draft_path=draft_path, reference_path=reference_path, output_path=output_dir
   )

   ```
3. Alternatively You can run the script from the command line:
   ```
   poetry run python pdf_parser/process_pdf.py <path_to_pdf_file> [output_directory]
   ```
   For example:
   ```
   poetry run python pdf_parser/process_pdf.py pdfs/Dupin__V1\ \(4\).pdf results/pdf_parser
   ```
   This will process the PDF file at `pdfs/Dupin__V1 (4).pdf` and save the results in the `results/pdf_parser` directory.

4. ### To generate rules for a pair of PDFs
   ```
   python pdf_parser/process_pdf.py --draft <path_to_draft_pdf_file> --reference <path_to_reference_pdf_file> --output <output_directory>
   ```

## How it works

1. The `process_pdf_task` function copies the input PDF to a temporary directory for processing.

2. It then runs the Java-based PDF parser using the `pdf-parser.jar` file.

3. The parser extracts metadata and text information from the PDF and saves it as a JSON file.

4. The `process_json` function in `text_extraction.py` processes this JSON data to create a full text representation of the PDF content.

5. The results, including metadata and full text, are saved in the `results/pdf_parser` directory as a JSON file.

## Customization

You can customize the text extraction process by modifying the `create_full_text` function in `text_extraction.py`. This function determines how the extracted text is formatted and structured.

## Troubleshooting

If you encounter any issues:

1. Ensure that Java `openjdk@17` is installed and accessible from the command line.
2. Install the required packages with `poetry install`.
3. Check that the `pdf-parser.jar` file is present in the `pdf_parser` directory.
4. Verify that the input PDF file exists in the specified location.

For any errors during processing, check the logs printed to the console for more information.
