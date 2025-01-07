import re
import json

import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv
import os
from loguru import logger

load_dotenv()


def postgres_handler() -> psycopg2.extensions.connection:
    user = os.getenv("XTRAWEB_POSTGRES_USER")
    password = os.getenv("XTRAWEB_POSTGRES_PASSWORD")
    database = os.getenv("XTRAWEB_POSTGRES_DB")
    host = os.getenv("XTRAWEB_POSTGRES_HOST")
    port = os.getenv("XTRAWEB_POSTGRES_PORT")
    try:
        conn = psycopg2.connect(
            user=user,
            password=password,
            database=database,
            host=host,
            port=port,
        )
        return conn
    except OperationalError as e:
        logger.error("Error connecting to the database:", e)
        return None


def store_pdf_data(conn: psycopg2.extensions.connection, file_name: str, title: str, pdf_data: dict, uid: str = None):
    """
    Inserts or updates the pdf_data table with information about the PDF.

    Parameters:
    pdf_data (dict): A dictionary containing 'uid', 'title', 'structured_data', 'full_text', 'metadata', and 'section_positions'.
    """
    # Sanitize the title
    sanitized_title = re.sub(r"\W+", "_", title).lower()
    file_name_ext = file_name.split(".")[-1]
    file_name_no_ext = file_name.split(".")[0]
    sanitized_file_name = re.sub(r"\W+", "_", file_name_no_ext).lower() + "." + file_name_ext
    # Prepare data
    structured_data = json.dumps(pdf_data.get("structured_data", {}))
    full_text = pdf_data.get("full_text", "")
    metadata = json.dumps(pdf_data.get("metadata", {}))
    error_msg = pdf_data.get("error", None)
    section_positions = json.dumps(pdf_data.get("section_positions", {}))

    # Update the insert query to include section_positions
    insert_pdf_data_query = """
    INSERT INTO pdf_data (uid, file_name, title, structured_data, full_text, metadata, error_msg, section_positions)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (uid) DO UPDATE
    SET file_name = EXCLUDED.file_name,
        title = EXCLUDED.title,
        structured_data = EXCLUDED.structured_data,
        full_text = EXCLUDED.full_text,
        metadata = EXCLUDED.metadata,
        error_msg = EXCLUDED.error_msg,
        section_positions = EXCLUDED.section_positions;
    """

    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    insert_pdf_data_query,
                    (
                        uid,
                        sanitized_file_name,
                        sanitized_title,
                        structured_data,
                        full_text,
                        metadata,
                        error_msg,
                        section_positions,
                    ),
                )
        logger.info("PDF data inserted successfully into pdf_data table")
        return sanitized_file_name, sanitized_title
    except Exception as e:
        logger.error(f"Error inserting PDF data: {e}")
        return "", "", ""


def store_sections_mapping(
    conn: psycopg2.extensions.connection, uid: str, file_name: str, title: str, sections_mapping: dict
):
    """
    Inserts or updates the taxonomy table with section mappings for a given PDF.

    Parameters:
    uid (str): Unique identifier for the PDF.
    title (str): Title of the PDF.
    sections_mapping (dict): A dictionary representing the mapping of sections to canonical names.
    """
    if uid == "":
        return
    # Convert sections_mapping to JSON
    sections_mapping_json = json.dumps(sections_mapping)
    error_msg = sections_mapping.get("error", None) if isinstance(sections_mapping, dict) else None

    # Updated insert query to include error_msg
    insert_taxonomy_query = """
    INSERT INTO taxonomy (uid, file_name, title, sections_mapping, error_msg)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (uid) DO UPDATE
    SET file_name = EXCLUDED.file_name,
        title = EXCLUDED.title,
        sections_mapping = EXCLUDED.sections_mapping,
        error_msg = EXCLUDED.error_msg;
    """

    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(insert_taxonomy_query, (uid, file_name, title, sections_mapping_json, error_msg))
        logger.info("Sections mapping inserted successfully into taxonomy table")
    except Exception as e:
        logger.error(f"Error inserting sections mapping: {e}")


def get_pdf_data_by_uid(conn: psycopg2.extensions.connection, uid: str) -> dict:
    """
    Retrieves PDF data from pdf_data table for a given UID.

    Parameters:
    conn (psycopg2.extensions.connection): Database connection
    uid (str): Unique identifier of the document

    Returns:
    dict: PDF document data
    """
    try:
        with conn:
            with conn.cursor() as cursor:
                pdf_query = """
                SELECT file_name, title, structured_data, full_text, metadata,
                       error_msg, section_positions
                FROM pdf_data
                WHERE uid = %s;
                """
                cursor.execute(pdf_query, (uid,))
                pdf_result = cursor.fetchone()

                if not pdf_result:
                    logger.warning(f"No PDF data found with UID: {uid}")
                    return None

                pdf_data = {
                    "uid": uid,
                    "file_name": pdf_result[0],
                    "title": pdf_result[1],
                    "structured_data": pdf_result[2] if pdf_result[2] else {},
                    "full_text": pdf_result[3],
                    "metadata": pdf_result[4] if pdf_result[4] else {},
                    "error_msg": pdf_result[5],
                    "section_positions": pdf_result[6] if pdf_result[6] else {},
                }

                return pdf_data

    except Exception as e:
        logger.error(f"Error retrieving PDF data: {e}")
        return None


def get_taxonomy_by_uid(conn: psycopg2.extensions.connection, uid: str) -> dict:
    """
    Retrieves taxonomy data from taxonomy table for a given UID.

    Parameters:
    conn (psycopg2.extensions.connection): Database connection
    uid (str): Unique identifier of the document

    Returns:
    dict: Taxonomy data
    """
    try:
        with conn:
            with conn.cursor() as cursor:
                taxonomy_query = """
                SELECT sections_mapping, error_msg
                FROM taxonomy
                WHERE uid = %s;
                """
                cursor.execute(taxonomy_query, (uid,))
                taxonomy_result = cursor.fetchone()

                if not taxonomy_result:
                    logger.warning(f"No taxonomy data found with UID: {uid}")
                    return None

                taxonomy_data = {
                    "uid": uid,
                    "sections_mapping": taxonomy_result[0] if taxonomy_result[0] else {},
                    "error_msg": taxonomy_result[1],
                }

                return taxonomy_data

    except Exception as e:
        logger.error(f"Error retrieving taxonomy data: {e}")
        return None
