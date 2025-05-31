import chromadb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorDB:
    def __init__(self, collection_name="social_phobia_info"):
        """
        Initializes the VectorDB client and collection.
        """
        try:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logging.info(f"Successfully connected to ChromaDB and got/created collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error initializing ChromaDB: {e}")
            self.collection = None # Ensure other methods handle this

    def add_texts(self, texts: list[str], ids: list[str] = None, metadatas: list[dict] = None):
        """
        Adds texts to the ChromaDB collection.

        Args:
            texts: A list of texts (strings) to add.
            ids: An optional list of corresponding IDs.
            metadatas: An optional list of corresponding metadatas.
        """
        if not self.collection:
            logging.error("Cannot add texts, collection is not initialized.")
            return

        try:
            if ids and metadatas:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            elif ids:
                self.collection.add(
                    documents=texts,
                    ids=ids
                )
            elif metadatas:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas
                )
            else:
                self.collection.add(
                    documents=texts
                )
            logging.info(f"Successfully added {len(texts)} documents to the collection.")
        except Exception as e:
            logging.error(f"Error adding texts to ChromaDB: {e}")

    def query_texts(self, query_text: str, n_results: int = 3):
        """
        Queries the collection for similar texts.

        Args:
            query_text: The text to query for.
            n_results: The number of results to return.

        Returns:
            A list of retrieved documents, or an empty-like structure on error.
        """
        empty_results = {'ids': [], 'distances': [], 'metadatas': [], 'documents': [[]], 'uris': [], 'data': []}
        if not self.collection:
            logging.error("Cannot query texts, collection is not initialized.")
            return empty_results

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            logging.info(f"Query successful for text: '{query_text}', found {len(results.get('documents', [[]])[0])} results.")
            return results
        except Exception as e:
            logging.error(f"Error querying texts from ChromaDB: {e}")
            return empty_results
