import os
from vector_db import VectorDB
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def populate_vector_db_from_file(vector_db_instance: VectorDB, file_path: str = "social_phobia_info.txt"):
    """
    Reads data from a file, processes it into chunks, and adds them to the VectorDB.

    Args:
        vector_db_instance: An initialized instance of VectorDB.
        file_path: The path to the text file containing the information.
    """
    logging.info(f"Starting to populate vector database from file: {file_path} using provided VectorDB instance.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found.")
        return
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        return

    # Process the text into manageable chunks (splitting by double newlines)
    chunks = content.split("\n\n")

    if not chunks or (len(chunks) == 1 and not chunks[0].strip()):
        logging.warning(f"No content found in '{file_path}' or content is empty after splitting.")
        return

    # Use the provided VectorDB instance
    if not vector_db_instance or not vector_db_instance.collection:
        logging.error("Provided VectorDB instance is invalid or its collection is not initialized. Cannot populate.")
        return

    # Prepare data for batch addition
    documents_to_add = []
    ids_to_add = []

    for i, chunk_text in enumerate(chunks):
        if chunk_text.strip():  # Ensure the chunk is not just whitespace
            doc_id = f"doc_chunk_{i}"
            documents_to_add.append(chunk_text.strip())
            ids_to_add.append(doc_id)

    if documents_to_add:
        try:
            vector_db_instance.add_texts(texts=documents_to_add, ids=ids_to_add)
            logging.info(f"Successfully processed and added {len(documents_to_add)} chunks to the vector database from '{file_path}'.")
        except Exception as e:
            logging.error(f"Error adding texts to VectorDB: {e}")
    else:
        logging.info(f"No valid chunks to add from '{file_path}'.")

    logging.info(f"Finished populating vector database from file: {file_path}")

if __name__ == "__main__":
    # Create a VectorDB instance specifically for direct script execution
    db_instance_for_script = VectorDB()
    if db_instance_for_script.collection is not None:
        populate_vector_db_from_file(vector_db_instance=db_instance_for_script)
    else:
        logging.error("Failed to initialize VectorDB for script execution. Data population aborted.")
