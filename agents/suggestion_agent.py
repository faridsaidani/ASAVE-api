# agents/suggestion_agent.py
from typing import List, Dict, Optional, Any
from langchain_core.output_parsers import JsonOutputParser
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)

class SuggestionAgent(BaseAgent):
    """
    AISGA: AI Suggestion Generation Agent.
    Generates clarifications for ambiguities or proposes enhancements/new clauses for gaps.
    Modified for API to accept string contexts and provide prompt details.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro-latest", temperature: float = 0.5):
        super().__init__(model_name=model_name, temperature=temperature,
                         system_message="You are an expert in drafting financial standards, specializing in AAOIFI FAS and Shari'ah compliance. Your suggestions must be precise, clear, and justifiable.")

    def _get_context_str(self, context_items: List[str], context_type: str = "Context") -> str:
        """Helper to format context strings."""
        if not context_items:
            return f"No specific {context_type} provided."
        # Each item in context_items is now assumed to be a string chunk
        return "\n\n---\n\n".join([f"{context_type} Excerpt:\n{item}" for item in context_items])

    def generate_clarification(self,
                               original_text: str,
                               identified_ambiguity: str,
                               fas_context_strings: List[str],
                               ss_context_strings: List[str]
                               ) -> Optional[Dict[str, Any]]:
        """
        Generates a revised text to clarify an identified ambiguity.
        Contexts are provided as lists of strings.
        Returns the full LLM output which includes the structured suggestion and prompt details.
        """
        fas_context_str = self._get_context_str(fas_context_strings, "FAS Context")
        ss_context_str = self._get_context_str(ss_context_strings, "SS Context")

        # Including a placeholder for prompt_details in the prompt itself, 
        # so the LLM is aware of this desired structure if it needs to generate it (though we'll add it post-hoc).
        prompt_template = """
        An ambiguity has been identified in a Financial Accounting Standard (FAS).
        Your task is to propose a clear and precise modification to the original text to resolve this ambiguity.
        Ensure your proposed text maintains the original intent where appropriate, enhances clarity, and aligns with Shari'ah principles.

        Original Text Snippet:
        ---
        {original_text}
        ---

        Identified Ambiguity:
        ---
        {identified_ambiguity}
        ---

        Relevant Context from the same FAS:
        ---
        {fas_context}
        ---

        Relevant Context from related Shari'ah Standards (SS):
        ---
        {ss_context}
        ---

        Based on all the above, provide your suggestion as a JSON object with the following exact keys:
        1.  "original_text": The original text snippet provided.
        2.  "proposed_text": Your revised, clarified text for the snippet.
        3.  "change_type": This should be "modification".
        4.  "reasoning": A detailed explanation of why your proposed text resolves the ambiguity and any other changes made.
        5.  "shariah_notes": Specific notes on how your proposed text aligns with Shari'ah principles, referencing the provided SS context or general Shari'ah knowledge if applicable. If no direct Shari'ah implication, state that.
        
        Ensure the output is a single, valid JSON object.

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        
        # Prepare details for prompt_details_actual
        prompt_vars_summary = {
            "original_text_len": len(original_text),
            "identified_ambiguity_len": len(identified_ambiguity),
            "num_fas_context_strings": len(fas_context_strings),
            "num_ss_context_strings": len(ss_context_strings),
            "fas_context_total_len": len(fas_context_str),
            "ss_context_total_len": len(ss_context_str),
        }
        # Full input data for the chain
        input_data_for_chain = {
            "original_text": original_text,
            "identified_ambiguity": identified_ambiguity,
            "fas_context": fas_context_str,
            "ss_context": ss_context_str
        }

        try:
            logger.info(f"Invoking AISGA generate_clarification for: '{original_text[:50]}...'")
            response = self.invoke_chain(chain, input_data_for_chain)
            
            if isinstance(response, dict) and all(k in response for k in ["original_text", "proposed_text", "change_type", "reasoning", "shariah_notes"]):
                # Add the actual prompt details to the response, not part of what LLM generates
                response["prompt_details_actual"] = {
                    "agent_method": "generate_clarification",
                    "prompt_template_version": "v1.api", # You can version your prompts
                    "variables_summary": prompt_vars_summary,
                    # "full_prompt_sent_to_llm": chain.prompt.format_prompt(**input_data_for_chain).to_string() # Potentially very long
                }
                logger.debug(f"AISGA clarification response: {response}")
                return response # Return the whole dict
            else:
                logger.warning(f"Unexpected response format from AISGA generate_clarification: {response}")
                # Attempt to return a structured error if possible
                return {
                    "error": "AISGA returned unexpected format.",
                    "llm_response": response, # Include what was received for debugging
                    "prompt_details_actual": { # Still provide prompt context
                         "agent_method": "generate_clarification",
                         "prompt_template_version": "v1.api",
                         "variables_summary": prompt_vars_summary
                    }
                }
        except Exception as e:
            logger.error(f"Error in AISGA generate_clarification: {e}", exc_info=True)
            return {
                "error": f"Exception in AISGA: {str(e)}",
                "prompt_details_actual": {
                     "agent_method": "generate_clarification",
                     "prompt_template_version": "v1.api",
                     "variables_summary": prompt_vars_summary
                }
            }

    def propose_enhancement_for_gap(self,
                                    gap_description: str,
                                    fas_name: str,
                                    fas_context_strings: List[str],
                                    ss_context_strings: List[str],
                                    external_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Proposes a new clause or enhancement to address an identified gap in an FAS.
        Contexts are provided as lists of strings.
        """
        fas_context_str = self._get_context_str(fas_context_strings, "FAS Context")
        ss_context_str = self._get_context_str(ss_context_strings, "SS Context")
        external_context_str = external_context if external_context else "No external context provided."

        prompt_template = """
        A potential gap or area for enhancement has been identified in {fas_name}.
        Your task is to draft a new clause or text to address this gap.
        The proposed text should be suitable for inclusion in an AAOIFI FAS, meaning it must be clear, precise, actionable, and Shari'ah compliant.

        Identified Gap/Enhancement Area:
        ---
        {gap_description}
        ---

        Relevant Context from {fas_name} (FAS):
        ---
        {fas_context}
        ---

        Relevant Context from related Shari'ah Standards (SS):
        ---
        {ss_context}
        ---

        Other Relevant External Context (if any):
        ---
        {external_context}
        ---

        Based on all the above, provide your proposal as a JSON object with the following exact keys:
        1.  "original_text": A brief description indicating the gap this insertion addresses (e.g., "Addressing the lack of guidance on digital asset accounting."). This key refers to the context of the gap, not a text to be modified.
        2.  "proposed_text": The complete text of the new clause or enhancement you are proposing.
        3.  "change_type": This should be "insertion".
        4.  "reasoning": A detailed explanation for why this new clause is needed, what it achieves, and how it fits into the existing standard structure (conceptually).
        5.  "shariah_notes": Specific notes on how the proposed new clause aligns with Shari'ah principles, referencing the provided SS context or general Shari'ah knowledge.
        
        Ensure the output is a single, valid JSON object.

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        
        prompt_vars_summary = {
            "gap_description_len": len(gap_description),
            "fas_name": fas_name,
            "num_fas_context_strings": len(fas_context_strings),
            "num_ss_context_strings": len(ss_context_strings),
            "external_context_len": len(external_context_str if external_context_str else ""),
            "fas_context_total_len": len(fas_context_str),
            "ss_context_total_len": len(ss_context_str),
        }
        input_data_for_chain = {
            "gap_description": gap_description,
            "fas_name": fas_name,
            "fas_context": fas_context_str,
            "ss_context": ss_context_str,
            "external_context": external_context_str
        }

        try:
            logger.info(f"Invoking AISGA propose_enhancement_for_gap for: '{gap_description[:50]}...'")
            response = self.invoke_chain(chain, input_data_for_chain)
            
            if isinstance(response, dict) and all(k in response for k in ["original_text", "proposed_text", "change_type", "reasoning", "shariah_notes"]):
                response["prompt_details_actual"] = {
                    "agent_method": "propose_enhancement_for_gap",
                    "prompt_template_version": "v1.api",
                    "variables_summary": prompt_vars_summary
                }
                logger.debug(f"AISGA enhancement response: {response}")
                return response
            else:
                logger.warning(f"Unexpected response format from AISGA propose_enhancement_for_gap: {response}")
                return {
                    "error": "AISGA returned unexpected format for enhancement proposal.",
                    "llm_response": response,
                    "prompt_details_actual": {
                         "agent_method": "propose_enhancement_for_gap",
                         "prompt_template_version": "v1.api",
                         "variables_summary": prompt_vars_summary
                    }
                }
        except Exception as e:
            logger.error(f"Error in AISGA propose_enhancement_for_gap: {e}", exc_info=True)
            return {
                "error": f"Exception in AISGA enhancement: {str(e)}",
                "prompt_details_actual": {
                     "agent_method": "propose_enhancement_for_gap",
                     "prompt_template_version": "v1.api",
                     "variables_summary": prompt_vars_summary
                }
            }