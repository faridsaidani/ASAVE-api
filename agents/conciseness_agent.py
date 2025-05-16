# agents/conciseness_agent.py
from typing import List, Dict, Any, Optional
from langchain_core.output_parsers import JsonOutputParser
from .base_agent import BaseAgent # Assuming base_agent.py is in the same directory
import logging
import os
import json

logger = logging.getLogger(__name__)

class ConcisenessAgent(BaseAgent):
    """
    A specialized AI agent focused on making text more concise
    while retaining essential meaning and Shari'ah compliance.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.2):
        """
        Initializes the ConcisenessAgent.

        Args:
            model_name (str): The LLM model to use (e.g., a faster model suitable for editing tasks).
            temperature (float): The creativity/determinism of the LLM. Lower for more focused editing.
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            system_message=(
                "You are an expert editor specializing in making financial and legal texts "
                "extremely concise and clear without losing essential meaning or Shari'ah compliance. "
                "Your primary goal is to reduce word count and simplify phrasing. "
                "Remove redundancy, jargon where possible (if clarity is maintained), and overly complex sentence structures. "
                "Always ensure the core message and any Shari'ah implications are preserved."
            )
        )
        self.agent_type = "Specialized_ConcisenessAgent" # For logging and identification

    def _get_context_str(self, context_items: List[str], context_type: str = "Context") -> str:
        """Helper to format context strings. (Copied from SuggestionAgent for consistency if needed)"""
        if not context_items:
            return f"No specific {context_type} provided for consideration."
        return "\n\n---\n\n".join([f"{context_type} Excerpt:\n{item}" for item in context_items])


    def make_concise(self,
                     text_to_make_concise: str,
                     shariah_context_strings: Optional[List[str]] = None, # To ensure conciseness doesn't violate Shari'ah
                     variant_name: str = "default" # For consistency if used in a multi-agent setup
                     ) -> Dict[str, Any]: # Returns a dict, including error structure if failed
        """
        Takes a text snippet and attempts to make it more concise.

        Args:
            text_to_make_concise (str): The input text.
            shariah_context_strings (Optional[List[str]]): Relevant Shari'ah principles or context
                                                           to guide the conciseness process and ensure compliance.
            variant_name (str): An optional name for this specific invocation/variant.

        Returns:
            Dict[str, Any]: A dictionary containing the original text, the proposed concise text,
                            reasoning, a Shari'ah compliance note, and prompt details.
                            Includes an "error" key if processing fails.
        """
        if not text_to_make_concise.strip():
            logger.warning(f"{self.agent_type}: Input text is empty or whitespace.")
            return {
                "error": "Input text is empty.",
                "original_text": text_to_make_concise,
                "proposed_text": text_to_make_concise, # Return original if input is bad
                "change_type": "no_change_due_to_empty_input",
                "reasoning": "Input text was empty, no changes made.",
                "shariah_notes": "N/A due to empty input.",
                "prompt_details_actual": {"agent_method": f"{self.agent_type}_{variant_name}_make_concise", "status": "empty_input"}
            }

        ss_context_str = self._get_context_str(shariah_context_strings or [], "Shari'ah Context")

        prompt_template = """
        The following "Original Text" needs to be made significantly more concise. Your goal is to reduce its length while preserving its core meaning and accurately reflecting its intent.
        Crucially, ensure that the conciseness changes maintain or enhance alignment with general Shari'ah principles and any specific "Relevant Shari'ah Context" provided.

        Original Text:
        ---
        {text_to_make_concise}
        ---

        Relevant Shari'ah Context (to guide conciseness and ensure compliance):
        ---
        {ss_context}
        ---
        
        Please provide your concise version as a JSON object with the following exact keys:
        1. "original_text": The original text exactly as provided above.
        2. "proposed_concise_text": Your significantly shortened and clarified version of the "Original Text". If the text is already perfectly concise, you may return the original text here.
        3. "reasoning_for_changes": Briefly explain the key changes you made to achieve conciseness (e.g., "removed redundant phrases like 'in order to'", "combined two sentences", "replaced verbose term X with Y"). If no changes were made because the text was already concise, state that.
        4. "shariah_compliance_note": A brief note confirming that your conciseness changes have maintained Shari'ah alignment, considering the provided context or general principles. If no specific Shari'ah context was provided, state that general principles were upheld.

        Ensure your output is a single, valid JSON object.

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        
        prompt_vars_summary = {
            "original_text_len": len(text_to_make_concise),
            "shariah_context_len": len(ss_context_str)
        }
        input_data_for_chain = {
            "text_to_make_concise": text_to_make_concise,
            "ss_context": ss_context_str
        }
        
        agent_method_name = f"{self.agent_type}_{variant_name}_make_concise"

        try:
            logger.info(f"Invoking {agent_method_name} for: '{text_to_make_concise[:70]}...'")
            response = self.invoke_chain(chain, input_data_for_chain)

            if isinstance(response, dict) and all(k in response for k in ["original_text", "proposed_concise_text", "reasoning_for_changes", "shariah_compliance_note"]):
                # Standardize output keys to match the general "suggestion_payload" structure
                # that the API orchestrator might expect (e.g., from AISGA)
                standardized_response = {
                    "original_text": response["original_text"],
                    "proposed_text": response["proposed_concise_text"],
                    "change_type": "conciseness_edit", # Specific type for this agent
                    "reasoning": response["reasoning_for_changes"],
                    "shariah_notes": response["shariah_compliance_note"], # Map to shariah_notes
                    "prompt_details_actual": {
                        "agent_method": agent_method_name,
                        "llm_model_used": self.model_name,
                        "temperature": self.llm.temperature,
                        "system_message_summary": self.system_message[:100]+"..." if self.system_message else "Default",
                        "variables_summary": prompt_vars_summary
                    }
                }
                return standardized_response
            else:
                logger.warning(f"Unexpected response format from {agent_method_name}: {response}")
                return {
                    "error": f"{agent_method_name} returned unexpected format.",
                    "llm_response_received": response, 
                    "prompt_details_actual": {
                        "agent_method": agent_method_name, "variables_summary": prompt_vars_summary
                    }
                }
        except Exception as e:
            logger.error(f"Error in {agent_method_name}: {e}", exc_info=True)
            return {
                "error": f"Exception in {agent_method_name}: {str(e)}",
                "prompt_details_actual": {
                    "agent_method": agent_method_name, "variables_summary": prompt_vars_summary
                }
            }

if __name__ == '__main__':
    # Example Usage (requires GOOGLE_API_KEY to be set in environment)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set the GOOGLE_API_KEY environment variable to run this example.")
    else:
        print("Running ConcisenessAgent example...")
        concise_agent = ConcisenessAgent()

        sample_text_verbose = (
            "In consideration of the fact that the aforementioned financial instrument is designed with the primary objective of "
            "facilitating liquidity management for Islamic financial institutions, it is therefore incumbent upon the "
            "structuring party to ensure that all contractual undertakings are fully compliant with the established "
            "tenets of Shari'ah, thereby avoiding any and all elements that could potentially be construed as Riba (interest) "
            "or Gharar (excessive uncertainty)."
        )
        sample_shariah_context = [
            "Riba (interest) in all its forms is strictly prohibited.",
            "Contracts must be clear and free from excessive uncertainty (Gharar) that could lead to disputes."
        ]

        print(f"\nOriginal Verbose Text:\n{sample_text_verbose}")
        
        concise_result = concise_agent.make_concise(
            text_to_make_concise=sample_text_verbose,
            shariah_context_strings=sample_shariah_context
        )

        print("\nConciseness Agent Result:")
        if "error" in concise_result:
            print(f"  Error: {concise_result['error']}")
            if "llm_response_received" in concise_result:
                print(f"  LLM Response (if any): {concise_result['llm_response_received']}")
        else:
            print(f"  Original Text Provided: {concise_result.get('original_text')}")
            print(f"  Proposed Concise Text: {concise_result.get('proposed_text')}")
            print(f"  Change Type: {concise_result.get('change_type')}")
            print(f"  Reasoning for Changes: {concise_result.get('reasoning')}")
            print(f"  Shari'ah Notes: {concise_result.get('shariah_notes')}")
        
        print("\nPrompt Details:")
        print(json.dumps(concise_result.get("prompt_details_actual", {}), indent=2))

        print("\n--- Example with already concise text ---")
        already_concise_text = "Assets are resources controlled by the entity."
        concise_result_2 = concise_agent.make_concise(
            text_to_make_concise=already_concise_text,
            shariah_context_strings=[] # No specific context this time
        )
        print("\nConciseness Agent Result (already concise):")
        if "error" in concise_result_2:
             print(f"  Error: {concise_result_2['error']}")
        else:
            print(f"  Proposed Concise Text: {concise_result_2.get('proposed_text')}")
            print(f"  Reasoning for Changes: {concise_result_2.get('reasoning')}")