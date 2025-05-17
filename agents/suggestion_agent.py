# agents/suggestion_agent.py
import json
import os
from typing import List, Dict, Optional, Any
from langchain_core.output_parsers import JsonOutputParser
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)

class SuggestionAgent(BaseAgent):
    """
    AISGA: AI Suggestion Generation Agent.
    Generates clarifications for ambiguities or proposes enhancements/new clauses for gaps.
    Includes a self-assessed confidence score for its suggestions.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro-latest", temperature: float = 0.5, system_message: str = None): # Adjusted default model
        super().__init__(model_name=model_name, temperature=temperature,
                         system_message= system_message or "You are an expert financial analyst specializing in AAOIFI standards. Your task is to generate clarifications for ambiguities or propose enhancements for gaps in the provided text segments, and to assess your confidence in each suggestion.")
        self.agent_type = "AISGA_SuggestionAgent"


    def _get_context_str(self, context_items: List[str], context_type: str = "Context") -> str:
        """Helper to format context strings."""
        if not context_items:
            return f"No specific {context_type} provided."
        return "\n\n---\n\n".join([f"{context_type} Excerpt:\n{item}" for item in context_items])

    def generate_clarification(self,
                               original_text: str,
                               identified_ambiguity: str,
                               fas_context_strings: List[str],
                               ss_context_strings: List[str],
                               variant_name: str = "default_clarification" # For prompt details
                               ) -> Optional[Dict[str, Any]]:
        """
        Generates a revised text to clarify an identified ambiguity.
        Contexts are provided as lists of strings.
        Returns the full LLM output which includes the structured suggestion, prompt details, and confidence score.
        """
        fas_context_str = self._get_context_str(fas_context_strings, "FAS Context")
        ss_context_str = self._get_context_str(ss_context_strings, "SS Context")

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
        4.  "reasoning": A detailed explanation of why your proposed text resolves the ambiguity and any other changes made. Include emojis if they enhance readability (e.g., ✅ for alignment, ⚠️ for caution).
        5.  "shariah_notes": Specific notes on how your proposed text aligns with Shari'ah principles, referencing the provided SS context or general Shari'ah knowledge if applicable. If no direct Shari'ah implication, state that.
        6.  "confidence_score": An integer between 0 and 100 representing your confidence that this proposed text is an optimal, compliant, and clear resolution to the identified ambiguity, given all context. 100 is highest confidence.
        
        Ensure the output is a single, valid JSON object.

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        
        prompt_vars_summary = {
            "original_text_len": len(original_text),
            "identified_ambiguity_len": len(identified_ambiguity),
            "num_fas_context_strings": len(fas_context_strings),
            "num_ss_context_strings": len(ss_context_strings),
        }
        input_data_for_chain = {
            "original_text": original_text,
            "identified_ambiguity": identified_ambiguity,
            "fas_context": fas_context_str,
            "ss_context": ss_context_str
        }
        agent_method_name = f"{self.agent_type}_{variant_name}_generate_clarification"

        try:
            logger.info(f"Invoking {agent_method_name} for: '{original_text[:50]}...'")
            response = self.invoke_chain(chain, input_data_for_chain)
            
            if isinstance(response, dict) and all(k in response for k in ["original_text", "proposed_text", "change_type", "reasoning", "shariah_notes", "confidence_score"]):
                try:
                    response["confidence_score"] = int(response["confidence_score"])
                except ValueError:
                    logger.warning(f"Could not parse confidence_score '{response['confidence_score']}' as int. Defaulting to 0.")
                    response["confidence_score"] = 0 # Default if LLM fails to provide valid int

                response["prompt_details_actual"] = {
                    "agent_method": agent_method_name,
                    "llm_model_used": self.model_name,
                    "temperature": self.llm.temperature,
                    "system_message_summary": self.system_message[:100]+"..." if self.system_message else "Default",
                    "variables_summary": prompt_vars_summary
                }
                logger.debug(f"AISGA clarification response: {response}")
                return response
            else:
                logger.warning(f"Unexpected response format from {agent_method_name}: {response}")
                return {
                    "error": f"{self.agent_type} returned unexpected format for clarification.",
                    "llm_response_received": response, 
                    "prompt_details_actual": { "agent_method": agent_method_name, "variables_summary": prompt_vars_summary }
                }
        except Exception as e:
            logger.error(f"Error in {agent_method_name}: {e}", exc_info=True)
            return {
                "error": f"Exception in {agent_method_name}: {str(e)}",
                "prompt_details_actual": { "agent_method": agent_method_name, "variables_summary": prompt_vars_summary }
            }

    def propose_enhancement_for_gap(self,
                                    gap_description: str,
                                    fas_name: str,
                                    fas_context_strings: List[str],
                                    ss_context_strings: List[str],
                                    external_context: Optional[str] = None,
                                    variant_name: str = "default_enhancement" # For prompt details
                                    ) -> Optional[Dict[str, Any]]:
        """
        Proposes a new clause or enhancement to address an identified gap in an FAS.
        Includes a self-assessed confidence score.
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
        4.  "reasoning": A detailed explanation for why this new clause is needed, what it achieves, and how it fits into the existing standard structure (conceptually). Include emojis if they enhance readability (e.g., ✅ for alignment, 💡 for idea).
        5.  "shariah_notes": Specific notes on how the proposed new clause aligns with Shari'ah principles, referencing the provided SS context or general Shari'ah knowledge.
        6.  "confidence_score": An integer between 0 and 100 representing your confidence that this proposed text is an optimal, compliant, and clear solution for the identified gap, given all context. 100 is highest confidence.
        
        Ensure the output is a single, valid JSON object.

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        
        prompt_vars_summary = {
            "gap_description_len": len(gap_description),
            "fas_name": fas_name,
            # ... other lengths if needed
        }
        input_data_for_chain = {
            "gap_description": gap_description,
            "fas_name": fas_name,
            "fas_context": fas_context_str,
            "ss_context": ss_context_str,
            "external_context": external_context_str
        }
        agent_method_name = f"{self.agent_type}_{variant_name}_propose_enhancement"

        try:
            logger.info(f"Invoking {agent_method_name} for: '{gap_description[:50]}...'")
            response = self.invoke_chain(chain, input_data_for_chain)
            
            if isinstance(response, dict) and all(k in response for k in ["original_text", "proposed_text", "change_type", "reasoning", "shariah_notes", "confidence_score"]):
                try:
                    response["confidence_score"] = int(response["confidence_score"])
                except ValueError:
                    logger.warning(f"Could not parse confidence_score '{response['confidence_score']}' as int for enhancement. Defaulting to 0.")
                    response["confidence_score"] = 0

                response["prompt_details_actual"] = {
                    "agent_method": agent_method_name,
                    "llm_model_used": self.model_name,
                    "temperature": self.llm.temperature,
                    "system_message_summary": self.system_message[:100]+"..." if self.system_message else "Default",
                    "variables_summary": prompt_vars_summary
                }
                logger.debug(f"AISGA enhancement response: {response}")
                return response
            else:
                logger.warning(f"Unexpected response format from {agent_method_name}: {response}")
                return {
                    "error": f"{self.agent_type} returned unexpected format for enhancement.",
                    "llm_response_received": response,
                    "prompt_details_actual": { "agent_method": agent_method_name, "variables_summary": prompt_vars_summary }
                }
        except Exception as e:
            logger.error(f"Error in {agent_method_name}: {e}", exc_info=True)
            return {
                "error": f"Exception in {agent_method_name}: {str(e)}",
                "prompt_details_actual": { "agent_method": agent_method_name, "variables_summary": prompt_vars_summary }
            }

# Example Usage (if run directly)
if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set the GOOGLE_API_KEY environment variable to run this example.")
    else:
        print("Running SuggestionAgent (AISGA) example with confidence score...")
        agent = SuggestionAgent(model_name="gemini-1.5-flash-latest") # Use Flash for faster example

        sample_original_text = "The company's assets may be utilized for permissible activities."
        sample_ambiguity = "The term 'permissible activities' is vague and lacks specific criteria, potentially leading to misinterpretation regarding Shari'ah compliance."
        sample_fas_context = ["FAS 1: General Presentation and Disclosure. Section 3.2: All activities must comply with Shari'ah.", "Appendix A: List of prohibited activities includes gambling and conventional interest-based lending."]
        sample_ss_context = ["Shari'ah Standard No. 5: Guarantees. Paragraph 12: Guarantees cannot be provided for activities that are themselves non-compliant.", "SS No. 1: Trading in Currencies. Paragraph 4: Speculative currency trading is not permitted."]

        clarification_result = agent.generate_clarification(
            original_text=sample_original_text,
            identified_ambiguity=sample_ambiguity,
            fas_context_strings=sample_fas_context,
            ss_context_strings=sample_ss_context
        )
        print("\n--- Clarification Result ---")
        print(json.dumps(clarification_result, indent=2))

        sample_gap = "The current FAS on Financial Instruments (FAS 25) does not explicitly address accounting for complex crypto-derivatives used for hedging purposes, particularly their valuation and Shari'ah compliance assessment."
        enhancement_result = agent.propose_enhancement_for_gap(
            gap_description=sample_gap,
            fas_name="FAS 25",
            fas_context_strings=["FAS 25, Para 10: Derivatives used for hedging must be effective.", "FAS 1: General principles of transparency."],
            ss_context_strings=["SS No. X: Hedging. Principle: Hedging actual risks is permissible. Speculation is not.", "SS No. Y: Financial Engineering. Principle: Complexity should not obscure Shari'ah non-compliance."]
        )
        print("\n--- Enhancement Proposal Result ---")
        print(json.dumps(enhancement_result, indent=2))