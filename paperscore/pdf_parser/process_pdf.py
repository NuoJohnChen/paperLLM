import os
import subprocess
import json
import shutil
import tempfile
from loguru import logger
import openai
import uuid
from dotenv import load_dotenv
from pdf_parser.postgres_handler import postgres_handler, store_pdf_data, store_sections_mapping

# from generate_rules import generate_rules

from pdf_parser.text_extraction import (
    process_json,
    extract_sections,
    extract_title_from_metadata,
    mapping_sections_with_llm,
)


# logging.basicConfig(
#     filename="PdfParser.log", level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
# )
# clear the content of the log file first
if os.path.exists("PdfParser.log"):
    with open("PdfParser.log", "w") as f:
        f.truncate()


def process_pdf_task(
    file_path,
    uid: str = None,
    run_section_extraction: bool = True,
    output_path: str = None,
    write_to_db: bool = True,
):
    try:
        # Use a secure temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the file to the temp_dir
            input_filename = os.path.basename(file_path)
            input_path = os.path.join(temp_dir, input_filename)
            shutil.copyfile(file_path, input_path)
            logger.info(f"Copied file to {input_path}")

            # Define prefixes for output files
            output_metadata_prefix = os.path.join(temp_dir, "data-")
            os.makedirs(output_metadata_prefix, exist_ok=True)
            # Log the current working directory and its contents
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Contents of current directory: {os.listdir()}")

            cmd = [
                "java",
                "-jar",
                "pdf_parser/pdf-parser.jar",
                input_path,
                "-g",
                output_metadata_prefix,
            ]
            # Run pdf-parser
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Log stdout and stderr
            if process.stdout:
                logger.info(f"pdf-parser stdout: {process.stdout}")
            if process.stderr:
                logger.error(f"pdf-parser stderr: {process.stderr}")

            if process.returncode != 0:
                raise Exception(f"pdf-parser processing failed: {process.stderr}")

            # Extract base filename without extension
            base_filename = os.path.splitext(input_filename)[0]

            # Construct the path to the output JSON file
            output_json_path = f"{output_metadata_prefix}{base_filename}.json"

            # Read and parse the metadata JSON file
            with open(output_json_path, "r") as f:
                metadata = json.load(f)
            if not uid:
                uid = str(uuid.uuid4())
            title = extract_title_from_metadata(input_path)
            full_text = process_json(metadata)
            structured_data = {"title": title}
            section_positions = {"title": 0}
            sections_mapping = {}
            if run_section_extraction:
                structured_data, section_positions = extract_sections(full_text)
                sections_mapping = mapping_sections_with_llm(structured_data)
                section_positions["title"] = 0
                structured_data["title"] = title
            full_text = title + "\n\n" + full_text
            logger.info(f"Extracted results for pdf {input_filename} with uid: {uid}")
            results = {
                "metadata": metadata,
                "structured_data": structured_data,
                "full_text": full_text,
                "sections_mapping": sections_mapping,
                "section_positions": section_positions,
            }

            if write_to_db:
                conn = postgres_handler()
                if conn:
                    cleaned_filename, title = store_pdf_data(
                        conn, uid=uid, file_name=input_filename, title=title, pdf_data=results
                    )
                    store_sections_mapping(
                        conn,
                        uid=uid,
                        file_name=cleaned_filename,
                        title=title,
                        sections_mapping=sections_mapping,
                    )
                    conn.close()

                else:
                    logger.error("Failed to connect to the database, please check your connection")
            # Save results to a local directory
            if output_path:
                filename = input_filename.replace(".pdf", ".json")
                result_path = os.path.join(output_path, filename)
                os.makedirs(output_path, exist_ok=True)
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=4)

            return results

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error during processing: {error_msg}")
        error_result = {"error": error_msg}

        # Write error to database if requested
        if write_to_db:
            try:
                conn = postgres_handler()
                if conn:
                    # Get basic file info even in error case
                    input_filename = os.path.basename(file_path)
                    title = extract_title_from_metadata(file_path) if os.path.exists(file_path) else input_filename

                    # Store error information
                    uid, cleaned_filename, title = store_pdf_data(conn, input_filename, title, error_result)
                    store_sections_mapping(conn, uid, cleaned_filename, title, {"error": error_msg})
                    conn.close()
            except Exception as db_error:
                logger.error(f"Error writing to database: {db_error}")

        return error_result


# def process_pdf_and_generate_rules(
#     client, draft_path: str, reference_path: str, output_path: str = None, write_to_db: bool = False
# ):
#     """
#     Process both draft and reference PDFs, then generate comparison rules.
#     """
#     try:
#         # Process both PDFs
#         logger.info(f"Processing draft PDF: {draft_path}")
#         draft_results = process_pdf_task(client, draft_path, output_path, write_to_db)

#         logger.info(f"Processing reference PDF: {reference_path}")
#         reference_results = process_pdf_task(client, reference_path, output_path, write_to_db)

#         # Generate rules using the processed results
#         logger.info("Generating comparison rules")
#         rules = generate_rules(client, draft_results, reference_results)

#         # Save rules if output path is provided
#         if output_path:
#             rules_filename = "comparison_rules.json"
#             rules_path = os.path.join(output_path, rules_filename)
#             with open(rules_path, "w") as f:
#                 json.dump(rules, f)
#             logger.info(f"Saved rules to {rules_path}")

#         return rules

#     except Exception as e:
#         logger.error(f"Error during processing and rule generation: {e}")
#         return {"error": str(e)}


if __name__ == "__main__":
    # Load environment variables and set up OpenAI client
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai.api_key)
    # import argparse

    # # Set up argument parser
    # parser = argparse.ArgumentParser(description="Process draft and reference PDFs to generate comparison rules.")
    # parser.add_argument("--draft", required=True, help="Path to the draft PDF file")
    # parser.add_argument("--reference", required=True, help="Path to the reference PDF file")
    # parser.add_argument("--output", default="pdf_parser/results", help="Output directory for results (optional)")

    # args = parser.parse_args()

    # Print success message
    # print(f"\nProcessing complete! Results saved to: {args.output}")

    # # Example usage
    draft_path = "pdfs/paper pairs v1/1to1_zhaomin_neurips/draft_zhaomin_fet_neurips24.pdf"
    # reference_path = "pdfs/paper pairs v1/1to2_fpga/reference_xinyu_fpgahls.pdf"

    # output_dir = "pdf_parser/results/pairing/zhaomin_neurips"

    # Only running PDF parser
    results = process_pdf_task(draft_path, write_to_db=False)
    print(results)
