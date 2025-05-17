# agents/contextual_update_agent.py
import os
from typing import Dict, Any, List
from langchain_core.output_parsers import JsonOutputParser
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)

class ContextualUpdateAgent(BaseAgent):
    """
    CUA: Contextual Update Agent.
    Analyzes new external information (news, guidelines) and its potential impact
    on an existing FAS document.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro-latest", temperature: float = 0.4):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            system_message=(
                "You are an expert financial analyst and regulatory specialist, skilled at "
                "understanding new information (like news, guidelines, or research) and "
                "assessing its impact on existing financial accounting standards (FAS)."
                "Your goal is to identify which parts of an FAS document might need review or updates "
                "based on the new contextual information provided."
            )
        )
        self.agent_type = "ContextualUpdateAgent_CUA"

    def analyze_impact(self,
                       new_context_text: str,
                       fas_document_content: str,
                       fas_document_id: str) -> Dict[str, Any]:
        """
        Analyzes the new context against the FAS document content.

        Args:
            new_context_text (str): The new external information.
            fas_document_content (str): The full text content of the target FAS document.
            fas_document_id (str): The ID/name of the FAS document.

        Returns:
            Dict[str, Any]: A structured analysis including:
                - summary_of_new_context: AI's understanding of the new info.
                - potential_impact_areas: List of dicts, each with:
                    - "fas_excerpt_guess": A snippet from fas_document_content likely affected.
                    - "reason_for_impact": Why this section might be affected.
                    - "suggested_action_type": e.g., "Review for Modification", "Consider New Clause", "Add Clarification Note".
                    - "key_points_from_new_context": Specific elements from new_context_text driving this.
                - overall_assessment: A brief on the significance of the new context for the FAS.
                - error: If an error occurred.
        """
        if not new_context_text.strip() or not fas_document_content.strip():
            logger.warning(f"{self.agent_type}: New context or FAS content is empty.")
            return {"error": "New contextual information or FAS document content is empty."}

        # For very long FAS documents, you might pass only a summary or use RAG to find relevant
        # FAS sections based on the new_context_text first, then pass those.
        # For now, assuming fas_document_content is manageable for the LLM context window.
        # Truncate if too long as a safety measure.
        max_fas_content_len = 70000 # Adjust based on model
        if len(fas_document_content) > max_fas_content_len:
            logger.warning(f"FAS document content for CUA is very long ({len(fas_document_content)} chars), truncating to {max_fas_content_len}.")
            fas_document_content_for_prompt = fas_document_content[:max_fas_content_len] + "\n... [CONTENT TRUNCATED] ..."
        else:
            fas_document_content_for_prompt = fas_document_content


        prompt_template = """
        You are tasked with analyzing new contextual information and its potential impact on a specific Financial Accounting Standard (FAS).

        New Contextual Information:
        ---
        {new_context_text}
        ---

        Existing FAS Document Content (ID: {fas_document_id}):
        ---
        {fas_document_content_for_prompt}
        ---

        Based on the "New Contextual Information", please perform the following analysis regarding the "Existing FAS Document Content":

        1.  **Summary of New Context:** Briefly summarize the key takeaways from the "New Contextual Information".
        2.  **Potential Impact Areas on FAS:** Identify specific sections or concepts within the "Existing FAS Document Content" that might be affected by the "New Contextual Information". For each area:
            a.  Provide a short "fas_excerpt_guess" (a direct quote or close paraphrase of a few lines from the FAS document that seems most relevant). If the impact suggests a new section is needed where none exists, state "No direct existing excerpt, consider new section related to [topic]".
            b.  Explain the "reason_for_impact" (why this part of the FAS might need review or change due to the new context).
            c.  Suggest a "suggested_action_type" (e.g., "Review for Modification", "Consider New Clause Addition", "Add Clarification Note", "No Direct Impact but Monitor").
            d.  List the "key_points_from_new_context" that specifically trigger this potential impact.
        3.  **Overall Assessment:** Provide a brief "overall_assessment" on the significance of the "New Contextual Information" for the FAS document (e.g., "High Impact - Requires Urgent Review", "Medium Impact - Further Analysis Recommended", "Low Impact - Minor Considerations").

        Provide your complete analysis as a single JSON object with the following exact keys:
        "summary_of_new_context", "potential_impact_areas" (which is a list of objects, each object having "fas_excerpt_guess", "reason_for_impact", "suggested_action_type", "key_points_from_new_context"), and "overall_assessment".

        JSON Output:
        """
        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        
        input_data = {
            "new_context_text": new_context_text,
            "fas_document_id": fas_document_id,
            "fas_document_content_for_prompt": fas_document_content_for_prompt
        }
        
        agent_method_name = f"{self.agent_type}_analyze_impact"
        prompt_details = { # For potential future inclusion in response if needed by API orchestrator
            "agent_method": agent_method_name,
            "llm_model_used": self.model_name,
            "temperature": self.llm.temperature,
            "input_summary": {
                "new_context_len": len(new_context_text),
                "fas_doc_id": fas_document_id,
                "fas_content_len_for_prompt": len(fas_document_content_for_prompt)
            }
        }

        try:
            logger.info(f"Invoking {agent_method_name} for FAS '{fas_document_id}' with new context (len: {len(new_context_text)}).")
            response = self.invoke_chain(chain, input_data)

            if isinstance(response, dict) and all(k in response for k in ["summary_of_new_context", "potential_impact_areas", "overall_assessment"]):
                response["_prompt_details_debug"] = prompt_details # Add debug info if needed later
                return response
            else:
                logger.warning(f"Unexpected response format from {agent_method_name}: {response}")
                return {
                    "error": f"{agent_method_name} returned unexpected format.",
                    "llm_response_received": response,
                    "_prompt_details_debug": prompt_details
                }
        except Exception as e:
            logger.error(f"Error in {agent_method_name}: {e}", exc_info=True)
            return {
                "error": f"Exception in {agent_method_name}: {str(e)}",
                "_prompt_details_debug": prompt_details
            }

if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set the GOOGLE_API_KEY environment variable to run this example.")
    else:
        print("Running ContextualUpdateAgent example...")
        cua_agent = ContextualUpdateAgent()

        sample_new_context = """
        Recent regulatory changes by the Central Bank (CB-FIN-2024-07) mandate new disclosure 
        requirements for digital asset holdings by Islamic Financial Institutions (IFIs). 
        Specifically, IFIs must now categorize digital assets based on their underlying Shari'ah contracts 
        (e.g., asset-backed tokens vs. utility tokens) and provide detailed risk assessments 
        for each category. This is effective from Q1 2025.
        """

        sample_fas_content = """
        Financial Accounting Standard 17: Investments
        ...
        Paragraph 25: Investments shall be classified as held-for-trading, available-for-sale, or held-to-maturity.
        Paragraph 26: Disclosure of investment valuation methods is required.
        ...
        Paragraph 40: For complex financial instruments, detailed notes explaining their nature and risks must be provided.
        ...
        (This FAS currently has no specific mention of 'digital assets' or 'cryptocurrencies')
        """
        sample_fas_id = "FAS-17 Rev.2022"

        analysis_result = cua_agent.analyze_impact(
            new_context_text=sample_new_context,
            fas_document_content=sample_fas_content,
            fas_document_id=sample_fas_id
        )

        print("\nContextual Update Agent Analysis Result:")
        import json
        print(json.dumps(analysis_result, indent=2))