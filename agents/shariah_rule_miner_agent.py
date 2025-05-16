# agents/shariah_rule_miner_agent.py
import os
import json
import uuid
from typing import List, Dict, Any, Optional
from langchain_core.output_parsers import JsonOutputParser
from .base_agent import BaseAgent
import logging
# from utils.document_processor import DocumentProcessor # Type hint if needed

logger = logging.getLogger(__name__)

class ShariahRuleMinerAgent(BaseAgent):
    """
    SRMA: Shari'ah Rule Miner Agent.
    Extracts and structures Shari'ah rules from AAOIFI Shari'ah Standards (SS) documents.
    """
    def __init__(self, model_name: str = "gemini-2.5-pro-preview-05-06", temperature: float = 0.3):
        super().__init__(model_name=model_name, temperature=temperature,
                         system_message="You are an expert in Islamic jurisprudence and AAOIFI Shari'ah Standards. Your task is to meticulously extract and structure Shari'ah rules from provided texts.")

    def _extract_potential_rules_from_chunk(self, text_chunk: str, standard_name: str) -> List[str]:
        """
        Extracts potential Shari'ah rule statements from a text chunk.
        """
        prompt_template = """
        From the following text chunk, taken from the AAOIFI Shari'ah Standard '{standard_name}',
        identify and extract sentences or short paragraphs that state a Shari'ah rule, prohibition, permission, or principle.
        Focus on prescriptive statements (e.g., "It is not permissible...", "It is necessary that...", "X is prohibited/allowed if...").
        List each potential rule text directly. Provide the output as a JSON list of strings.
        If no rule-like statements are found, return an empty list.

        Text Chunk:
        ---
        {text_chunk}
        ---

        JSON List of Potential Rule Texts:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        try:
            response = self.invoke_chain(chain, {"text_chunk": text_chunk, "standard_name": standard_name})
            if isinstance(response, list) and all(isinstance(item, str) for item in response):
                return response
            elif isinstance(response, dict) and "rules" in response and isinstance(response["rules"], list): # Handle if LLM wraps in a dict
                return response["rules"]
            logger.warning(f"Unexpected response type for potential rules: {type(response)}. Expected list of strings. Standard: {standard_name}, Chunk: {text_chunk[:50]}")
            return []
        except Exception as e:
            logger.error(f"Error in _extract_potential_rules_from_chunk for '{standard_name}': {e} (Chunk: {text_chunk[:50]})", exc_info=True)
            return []

    def _format_rule_to_json_structure(self,
                                       rule_text: str,
                                       standard_name_full: str,
                                       standard_short_code: str,
                                       chunk_page_content_for_ref: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Takes a raw rule text and formats it into the specified JSON structure using an LLM call.
        """
        context_for_ref = f"The rule was extracted from a context similar to: '{chunk_page_content_for_ref[:300]}...'" if chunk_page_content_for_ref else ""

        prompt_template = f"""
        Given the following Shari'ah rule text, extracted from '{standard_name_full}' (short code: {standard_short_code}):
        Rule Text: "{rule_text}"
        {context_for_ref}

        Please structure this rule into a JSON object with the following exact keys:
        1.  "rule_id": (This will be generated externally, do not include in LLM output)
        2.  "standard_ref": The full name of the Shari'ah Standard it originates from (e.g., "{standard_name_full}").
        3.  "principle_keywords": A list of 3-5 relevant Shari'ah keywords or concepts (e.g., ["Riba", "Gharar", "Maysir", "Contract", "Permissibility"]).
        4.  "description": A concise, clear, and complete statement of the rule based on the provided rule text. Rephrase for clarity if necessary, but maintain original meaning.
        5.  "validation_query_template": A question template that can be used to validate if a financial product or transaction complies with this rule.
            This template should include placeholders like '{{{{suggestion_text}}}}' or '{{{{product_details}}}}' and '{{{{specific_aspect}}}}'.
            Example: "Does the proposed transaction ('{{{{suggestion_text}}}}') regarding '{{{{specific_aspect}}}}' involve any element of Riba Al-Fadl as per this rule? Analyze based on the principle of equal countervalues in exchanges of ribawi items."

        Ensure your output is a single, valid JSON object with only the keys: "standard_ref", "principle_keywords", "description", "validation_query_template".

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        try:
            # We construct the input for the prompt, not passing rule_id as it's generated by Python code
            response = self.invoke_chain(chain, {
                # "rule_text": rule_text, # Already in prompt
                # "standard_name_full": standard_name_full, # Already in prompt
                # "standard_short_code": standard_short_code, # Already in prompt
                # "chunk_page_content_for_ref": chunk_page_content_for_ref # Already in prompt
            }) # Input is embedded in the prompt_template string

            if isinstance(response, dict) and all(k in response for k in ["standard_ref", "principle_keywords", "description", "validation_query_template"]):
                formatted_rule = {
                    "rule_id": f"{standard_short_code.upper()}-{str(uuid.uuid4())[:8].upper()}", # Generate UUID here
                    "standard_ref": response.get("standard_ref", standard_name_full),
                    "principle_keywords": response.get("principle_keywords", []),
                    "description": response.get("description", rule_text),
                    "validation_query_template": response.get("validation_query_template", "")
                }
                return formatted_rule
            logger.warning(f"Unexpected response format for rule structuring: {response}. Rule text: {rule_text[:50]}")
            return None
        except Exception as e:
            logger.error(f"Error in _format_rule_to_json_structure: {e}. Rule text: {rule_text[:50]}", exc_info=True)
            return None


    def mine_rules_from_document_list(self,
                                      ss_documents_with_metadata: List[Dict[str, Any]],
                                      doc_processor_instance, # Instance of DocumentProcessor
                                      base_output_dir: str) -> str:
        """
        Processes multiple Shari'ah Standard (SS) PDFs, extracts rules, and saves them.
        Saves individual JSON files per standard and a combined JSON file.
        """
        os.makedirs(base_output_dir, exist_ok=True)
        all_mined_rules = []
        
        for ss_info in ss_documents_with_metadata:
            pdf_path = ss_info["filepath"]
            standard_name_full = ss_info["standard_name_full"]
            standard_short_code = ss_info["standard_short_code"]
            standard_rules = []
            
            logger.info(f"Processing Shari'ah Standard for SRMA: {standard_name_full} ({pdf_path})")
            print(f"[SRMA] Processing Shari'ah Standard: {standard_name_full} ({pdf_path})")
            
            try:
                print(f"[SRMA] Loading PDF: {pdf_path}")
                docs = doc_processor_instance.load_pdf(pdf_path)
                print(f"[SRMA] Chunking document: {pdf_path}")
                chunks = doc_processor_instance.chunk_text(docs, chunk_size=2000, chunk_overlap=400)
                print(f"[SRMA] Document chunked into {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Error loading/chunking {pdf_path} for SRMA: {e}", exc_info=True)
                print(f"[SRMA] ERROR: Failed to load/chunk {pdf_path}: {e}")
                continue

            for i, chunk in enumerate(chunks):
                logger.debug(f"  SRMA Processing chunk {i+1}/{len(chunks)} for {standard_short_code}...")
                print(f"[SRMA] Extracting potential rules from chunk {i+1}/{len(chunks)}...")
                potential_rule_texts = self._extract_potential_rules_from_chunk(chunk.page_content, standard_name_full)
                
                for rule_text in potential_rule_texts:
                    if len(rule_text.strip()) < 20 : continue # Skip very short, likely irrelevant matches
                    logger.debug(f"    SRMA Formatting potential rule: '{rule_text[:70]}...'")
                    print(f"[SRMA] Formatting rule: {rule_text[:60]}...")
                    formatted_rule = self._format_rule_to_json_structure(
                        rule_text,
                        standard_name_full,
                        standard_short_code,
                        chunk.page_content # Provide context for better formatting
                    )
                    if formatted_rule:
                        logger.info(f"      + SRMA Mined Rule ID: {formatted_rule['rule_id']} from {standard_short_code}")
                        print(f"[SRMA] + Mined Rule ID: {formatted_rule['rule_id']} from {standard_short_code}")
                        standard_rules.append(formatted_rule)
                        all_mined_rules.append(formatted_rule)
            
            if standard_rules:
                output_path_standard = os.path.join(base_output_dir, f"mined_rules_{standard_short_code}.json")
                try:
                    with open(output_path_standard, 'w', encoding='utf-8') as f:
                        json.dump(standard_rules, f, indent=2, ensure_ascii=False)
                    logger.info(f"  Saved {len(standard_rules)} rules for {standard_short_code} to {output_path_standard}")
                    print(f"[SRMA] Saved {len(standard_rules)} rules for {standard_short_code} to {output_path_standard}")
                except IOError as e:
                    logger.error(f"  Failed to save rules for {standard_short_code} to {output_path_standard}: {e}")
                    print(f"[SRMA] ERROR: Failed to save rules for {standard_short_code}: {e}")

        output_path_combined = os.path.join(base_output_dir, "shariah_rules_mined_combined.json")
        try:
            with open(output_path_combined, 'w', encoding='utf-8') as f:
                json.dump(all_mined_rules, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved a total of {len(all_mined_rules)} SRMA rules to {output_path_combined}")
            print(f"[SRMA] Saved a total of {len(all_mined_rules)} rules to {output_path_combined}")
        except IOError as e:
            logger.error(f"Failed to save combined SRMA rules to {output_path_combined}: {e}")
            print(f"[SRMA] ERROR: Failed to save combined rules: {e}")
            return "" # Return empty if save fails
        return output_path_combined