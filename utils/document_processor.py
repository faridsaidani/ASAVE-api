# utils/document_processor.py
import os
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles loading, chunking, embedding, and vector store creation/loading for documents.
    """
    def __init__(self, embedding_model_name: str = "models/embedding-001"):
        """
        Initializes the DocumentProcessor with a Google Generative AI Embeddings model.

        Args:
            embedding_model_name (str): The name of the embedding model to use.
        """
        if not os.getenv("GOOGLE_API_KEY"):
            # This check is good, but the API server will also check and handle API key.
            # Agent initialization will also fail if not set.
            logger.warning("GOOGLE_API_KEY environment variable not set during DocumentProcessor init.")
            # raise ValueError("GOOGLE_API_KEY environment variable not set.") # Avoid hard crash here, let higher level handle
        self.embedding_model_name = embedding_model_name
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
            logger.info(f"DocumentProcessor initialized with embedding model: {embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings with model {self.embedding_model_name}: {e}")
            raise ValueError(f"Failed to initialize embedding model. Check GOOGLE_API_KEY and model name. Error: {e}")


    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Loads a PDF file and returns its content as a list of Document objects.

        Args:
            pdf_path (str): The file path to the PDF.

        Returns:
            List[Document]: A list of Document objects, typically one per page.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
        return documents

    def chunk_text(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunks.

        Args:
            documents (List[Document]): The documents to chunk.
            chunk_size (int): The maximum size of each chunk (in characters).
            chunk_overlap (int): The overlap between consecutive chunks.

        Returns:
            List[Document]: A list of chunked Document objects.
        """
        logger.info(f"Chunking {len(documents)} documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks

    def create_vector_store(self, chunks: List[Document], persist_directory: Optional[str] = None) -> Chroma:
        """
        Creates a ChromaDB vector store from document chunks.

        Args:
            chunks (List[Document]): The document chunks to add to the vector store.
            persist_directory (Optional[str]): Directory to persist the DB. If None, it's in-memory.

        Returns:
            Chroma: The created ChromaDB vector store.
        """
        if not chunks:
            logger.error("Cannot create vector store from empty list of chunks.")
            raise ValueError("Cannot create vector store from empty list of chunks.")

        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            logger.info(f"Persisting vector store to: {persist_directory}")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=persist_directory
            )
        else:
            logger.info("Creating in-memory vector store.")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model
            )
        logger.info("Vector store created successfully.")
        return vector_store

    def load_vector_store(self, persist_directory: str) -> Optional[Chroma]:
        """
        Loads a persisted ChromaDB vector store from disk.

        Args:
            persist_directory (str): The directory where the DB is persisted.

        Returns:
            Optional[Chroma]: The loaded ChromaDB vector store, or None if not found.
        """
        if not os.path.exists(persist_directory):
            logger.warning(f"Persistence directory not found: {persist_directory}. Cannot load vector store.")
            return None
        logger.info(f"Loading vector store from: {persist_directory}")
        try:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_model
            )
            logger.info("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store from {persist_directory}: {e}")
            return None