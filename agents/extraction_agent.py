# agents/extraction_agent.py
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)

class ExtractionAgent(BaseAgent):
    """
    KEEA: Knowledge Extraction & Enhancement Agent.
    Extracts definitions, key clauses, and identifies ambiguities from text.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.2):
        super().__init__(model_name=model_name, temperature=temperature, system_message="You are an expert financial analyst specializing in AAOIFI standards. Your task is to meticulously extract information from provided text segments.")

    def extract_definitions(self, text_chunk: str) -> List[Dict[str, str]]:
        """
        Uses Gemini to extract terms and their definitions from a given text chunk.
        """
        prompt_template = """
        From the following text, extract all clearly defined terms and their corresponding definitions.
        A term is usually presented in bold or followed by a phrase like "means", "is defined as", or similar.
        Provide the output as a JSON list of objects, where each object has a "term" key and a "definition" key.
        If no definitions are found, return an empty list.

        Text:
        ---
        {text_chunk}
        ---

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        try:
            response = self.invoke_chain(chain, {"text_chunk": text_chunk})
            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and response.get("definitions"): # Handle LLM wrapping
                 return response.get("definitions")
            logger.warning(f"Unexpected response type for definitions: {type(response)}. Expected list. Chunk: {text_chunk[:100]}")
            return []
        except Exception as e:
            logger.error(f"Error in extract_definitions: {e} for chunk: {text_chunk[:100]}", exc_info=True)
            return []

    def identify_key_clauses(self, topic: str, standard_name: str, fas_vector_store: Chroma, k: int = 5) -> List[str]:
        """
        Identifies key clauses related to a specific topic within a given FAS standard using RAG.
        """
        if not fas_vector_store:
            logger.error("FAS vector store not provided to identify_key_clauses.")
            return []
            
        logger.info(f"Identifying key clauses for topic '{topic}' in standard '{standard_name}'...")
        try:
            retriever = fas_vector_store.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.get_relevant_documents(f"Key clauses related to {topic} in {standard_name}")
            
            if not relevant_docs:
                logger.info(f"No relevant documents found for topic '{topic}'.")
                return []

            context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            prompt_template = """
            You are an expert in AAOIFI Financial Accounting Standards.
            Based on the following context from {standard_name}, identify and list the key clauses or statements related to the topic: "{topic}".
            Present each key clause clearly. If quoting directly, ensure accuracy. If summarizing, be concise and precise.
            If multiple distinct clauses are found, list them separately. Output as a JSON list of strings.

            Context:
            ---
            {context}
            ---

            JSON List of Key Clauses related to "{topic}":
            """
            chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
            response = self.invoke_chain(chain, {
                "topic": topic,
                "standard_name": standard_name,
                "context": context
            })
            if isinstance(response, list) and all(isinstance(item, str) for item in response):
                return response
            logger.warning(f"Unexpected response type for key_clauses: {type(response)}. Expected list of strings.")
            return [] # Fallback
        except Exception as e:
            logger.error(f"Error in identify_key_clauses: {e}", exc_info=True)
            return []

    def find_ambiguities(self, text_chunk: str) -> List[Dict[str, str]]:
        """
        Uses Gemini to pinpoint unclear phrasing or ambiguities in a given text chunk.
        """
        prompt_template = """
        Analyze the following text from an AAOIFI standard for any ambiguities, unclear phrasing, or statements
        that could lead to misinterpretation. For each ambiguity found, describe the ambiguous phrase and
        explain why it is ambiguous or could be misinterpreted.

        Provide the output as a JSON list of objects, where each object has an "ambiguous_phrase" key
        (the exact text snippet) and a "reason_for_ambiguity" key (your explanation).
        If no ambiguities are found, return an empty list.

        Text:
        ---
        {text_chunk}
        ---

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        try:
            response = self.invoke_chain(chain, {"text_chunk": text_chunk})
            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and response.get("ambiguities"): # Handle LLM wrapping
                 return response.get("ambiguities")
            logger.warning(f"Unexpected response type for ambiguities: {type(response)}. Expected list. Chunk: {text_chunk[:100]}")
            return []
        except Exception as e:
            logger.error(f"Error in find_ambiguities: {e} for chunk: {text_chunk[:100]}", exc_info=True)
            return []