# agents/validation_agent.py
import json
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from .base_agent import BaseAgent
import logging
import os

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """
    SCVA & ISCCA: Shari'ah Compliance Validation Agent & Inter-Standard Consistency Check Agent.
    Validates suggestions for Shari'ah compliance and (conceptually) inter-standard consistency.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro-latest", temperature: float = 0.1):
        super().__init__(model_name=model_name, temperature=temperature,
                         system_message="You are an expert Shari'ah scholar and AAOIFI standards auditor. Your task is to critically evaluate text for compliance and consistency based on provided rules and context.")

    def _load_explicit_shariah_rules(self, shariah_rules_path: str) -> List[Dict[str, Any]]:
        """Loads Shari'ah rules from a JSON file."""
        if not shariah_rules_path or not os.path.exists(shariah_rules_path):
            logger.warning(f"Shari'ah rules file not found or path not specified: {shariah_rules_path}")
            return []
        try:
            with open(shariah_rules_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            if not isinstance(rules, list):
                logger.warning(f"Shari'ah rules file content at {shariah_rules_path} is not a list.")
                return []
            logger.info(f"Loaded {len(rules)} rules from {shariah_rules_path}")
            return rules
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from Shari'ah rules file: {shariah_rules_path}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading Shari'ah rules from {shariah_rules_path}: {e}", exc_info=True)
            return []


    def validate_shariah_compliance(self,
                                    proposed_suggestion_object: Dict[str, Any],
                                    shariah_rules_explicit_path: str,
                                    ss_vector_store: Optional[Chroma],
                                    mined_shariah_rules_path: Optional[str] = None,
                                    k_semantic: int = 3) -> Dict[str, Any]:
        """
        Validates a proposed suggestion for Shari'ah compliance.
        """
        suggestion_text = proposed_suggestion_object.get("proposed_text", "")
        aisga_shariah_notes = proposed_suggestion_object.get("shariah_notes", "N/A")
        
        report = {
            "overall_status": "Needs Expert Review",
            "summary_explanation": "Initial assessment pending detailed checks.",
            "explicit_rule_checks": [],
            "semantic_validation_notes": "Not performed (SS Vector Store missing or other issue).",
            "conflicting_rules_identified": [],
            "debug_info": {
                "explicit_rules_path_used": shariah_rules_explicit_path,
                "mined_rules_path_used": mined_shariah_rules_path,
                "ss_vector_store_available": bool(ss_vector_store)
            }
        }

        if not suggestion_text:
            report["summary_explanation"] = "No proposed text provided for validation."
            report["overall_status"] = "Error"
            return report

        # 1. Validate against explicit Shari'ah rules
        all_rules_to_check = self._load_explicit_shariah_rules(shariah_rules_explicit_path)
        if mined_shariah_rules_path and os.path.exists(mined_shariah_rules_path):
            mined_rules = self._load_explicit_shariah_rules(mined_shariah_rules_path)
            if mined_rules:
                all_rules_to_check.extend(mined_rules)
                logger.info(f"Augmented with {len(mined_rules)} mined rules for validation.")
        
        if not all_rules_to_check:
            report["explicit_rule_checks"].append({
                "rule_id": "N/A", "status": "Skipped", "explanation": "No explicit or mined rules loaded for validation."
            })
        else:
            logger.info(f"SCVA: Validating against {len(all_rules_to_check)} explicit/mined Shari'ah rules...")
            for rule in all_rules_to_check:
                rule_id = rule.get("rule_id", "UnknownRule")
                description = rule.get("description", "No description")
                query_template = rule.get("validation_query_template")

                if not query_template:
                    report["explicit_rule_checks"].append({
                        "rule_id": rule_id, "status": "Skipped", "explanation": "Rule lacks a validation_query_template."
                    })
                    continue
                
                specific_aspect = "the proposed standard modification" # Could be more dynamic
                try:
                    validation_query = query_template.format(
                        suggestion_text=suggestion_text,
                        product_details=suggestion_text,
                        specific_aspect=specific_aspect
                    )
                except KeyError as e:
                    logger.warning(f"Invalid placeholder in query_template for rule {rule_id}: {e}")
                    report["explicit_rule_checks"].append({
                        "rule_id": rule_id, "status": "Error", "explanation": f"Invalid placeholder in query_template: {e}"
                    })
                    continue
                
                prompt = f"""
                Rule Description: {description} (Rule ID: {rule_id})
                Validation Query: {validation_query}

                Proposed Suggestion Text for AAOIFI Standard:
                ---
                {suggestion_text}
                ---
                Original Shari'ah Notes for this suggestion (from AISGA):
                ---
                {aisga_shariah_notes}
                ---

                Based ONLY on the Rule Description and the Validation Query, assess the "Proposed Suggestion Text".
                Is the suggestion compliant with this specific rule?

                Provide your assessment as a JSON object with keys:
                "status": ("Compliant", "Non-Compliant", "Potentially Non-Compliant", "Not Applicable")
                "explanation": "Brief reasoning for your status, strictly based on the rule."

                JSON Output:
                """
                chain = self._create_chain(prompt, output_parser=JsonOutputParser())
                try:
                    assessment = self.invoke_chain(chain, {}) # Input is in the prompt
                    status = assessment.get("status", "Error")
                    explanation = assessment.get("explanation", "LLM did not provide explanation.")
                    report["explicit_rule_checks"].append({
                        "rule_id": rule_id, "status": status, "explanation": explanation, "rule_description": description
                    })
                    if "Non-Compliant" in status: # Catches "Non-Compliant" and "Potentially Non-Compliant"
                        report["conflicting_rules_identified"].append(f"Rule ID: {rule_id} - Status: {status} - Desc: {description[:70]}...")
                except Exception as e_llm:
                    logger.error(f"LLM query failed for rule {rule_id}: {e_llm}", exc_info=True)
                    report["explicit_rule_checks"].append({
                        "rule_id": rule_id, "status": "Error", "explanation": f"LLM query failed: {str(e_llm)[:100]}"
                    })

        # 2. Semantic validation against SS vector store context
        semantic_notes_list = []
        if ss_vector_store:
            logger.info("SCVA: Performing semantic Shari'ah validation using SS vector store...")
            try:
                retriever = ss_vector_store.as_retriever(search_kwargs={"k": k_semantic})
                query_for_ss_context = f"Shari'ah implications, principles, and alignment for the following proposed text for an AAOIFI standard: '{suggestion_text}'. Consider general Shari'ah principles and specific concepts mentioned in the text or related to its domain. AISGA's original notes: '{aisga_shariah_notes}'"
                relevant_ss_docs = retriever.get_relevant_documents(query_for_ss_context)
                
                if relevant_ss_docs:
                    ss_context_str = "\n\n---\n\n".join([f"SS Excerpt (Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" for doc in relevant_ss_docs])
                    
                    semantic_prompt = f"""
                    Proposed Suggestion Text for AAOIFI Standard:
                    ---
                    {suggestion_text}
                    ---
                    Original Shari'ah Notes for this suggestion (from AISGA):
                    ---
                    {aisga_shariah_notes}
                    ---
                    Relevant Excerpts from Shari'ah Standards (SS) for context:
                    ---
                    {ss_context_str}
                    ---
                    Based on the provided SS excerpts, analyze the Shari'ah alignment of the "Proposed Suggestion Text".
                    Specifically comment on:
                    1. Consistency with the principles evident in the SS excerpts.
                    2. Any potential contradictions or areas of concern highlighted by the SS excerpts that the suggestion might raise.
                    3. Overall Shari'ah conduciveness OF THE SUGGESTION based ONLY on this provided context.

                    Provide a concise summary of your semantic analysis. If the excerpts are not relevant or don't offer insight, state that.
                    """
                    chain = self._create_chain(semantic_prompt, output_parser=StrOutputParser())
                    semantic_analysis_result = self.invoke_chain(chain, {})
                    report["semantic_validation_notes"] = semantic_analysis_result.strip()
                else:
                    report["semantic_validation_notes"] = "No relevant Shari'ah Standard excerpts found via semantic search for this suggestion."
            except Exception as e_semantic:
                error_msg = f"Error during semantic validation: {e_semantic}"
                logger.error(error_msg, exc_info=True)
                report["semantic_validation_notes"] = error_msg
        
        # 3. Determine overall_status and summary_explanation (Simplified Logic)
        # This can be made more sophisticated, potentially with another LLM call as originally designed.
        # For now, a rule-based summary:
        is_non_compliant_explicit = any("Non-Compliant" in check["status"] for check in report["explicit_rule_checks"])
        is_potentially_non_compliant_explicit = any("Potentially Non-Compliant" in check["status"] for check in report["explicit_rule_checks"])

        if is_non_compliant_explicit:
            report["overall_status"] = "Non-Compliant"
            report["summary_explanation"] = "Flagged as Non-Compliant by one or more explicit Shari'ah rule checks. Expert review mandatory."
        elif is_potentially_non_compliant_explicit:
            report["overall_status"] = "Needs Expert Review"
            report["summary_explanation"] = "Flagged as Potentially Non-Compliant by one or more explicit Shari'ah rule checks. Requires expert Shari'ah review."
        else: # No explicit non-compliance
            if "concern" in report["semantic_validation_notes"].lower() or \
               "contradiction" in report["semantic_validation_notes"].lower() or \
               "violates" in report["semantic_validation_notes"].lower() : # Basic check of semantic notes
                 report["overall_status"] = "Needs Expert Review"
                 report["summary_explanation"] = "Semantic analysis against Shari'ah Standards raised potential concerns or did not find strong support. Requires expert Shari'ah review."
            elif not report["conflicting_rules_identified"] and report["explicit_rule_checks"]: # All explicit checks are Compliant or N/A
                 report["overall_status"] = "Compliant (Initial Check)"
                 report["summary_explanation"] = "Initial automated checks suggest compliance based on available rules and semantic analysis. Final determination requires expert review."
            else: # Default if no strong signals
                 report["overall_status"] = "Needs Expert Review"
                 report["summary_explanation"] = "Further expert review is recommended to ascertain full Shari'ah compliance."

        if not report["explicit_rule_checks"] and "Not performed" in report["semantic_validation_notes"]:
            report["summary_explanation"] = "Validation could not be performed comprehensively (no rules loaded or no SS context found)."

        return report

    def validate_inter_standard_consistency(self,
                                           proposed_suggestion_object: Dict[str, Any],
                                           fas_name: str,
                                           all_fas_vector_store: Optional[Chroma],
                                           k: int = 3) -> Dict[str, Any]:
        """
        Checks for terminology and principle consistency against other FAS context.
        """
        report = {
            "status": "Not Performed",
            "explanation": "Inter-standard consistency check skipped (e.g., FAS vector store not available or other issue).",
            "conflicting_terms_or_principles": [],
            "relevant_other_fas_excerpts": "N/A"
        }

        if not all_fas_vector_store:
            logger.warning("ISCCA: All FAS vector store not provided. Skipping consistency check.")
            return report

        suggestion_text = proposed_suggestion_object.get("proposed_text", "")
        if not suggestion_text:
            report["explanation"] = "No suggestion text provided for ISCCA."
            return report
            
        logger.info(f"ISCCA: Performing inter-standard consistency check for suggestion in {fas_name}...")
        try:
            retriever = all_fas_vector_store.as_retriever(search_kwargs={"k": k})
            query = f"Review for consistency with other AAOIFI Financial Accounting Standards: Key terms, definitions, principles or clauses related to concepts in the following proposed text for {fas_name}: '{suggestion_text[:250]}...'. Highlight any potential contradictions or misalignments in terminology or underlying principles found in other FAS documents."
            
            relevant_docs = retriever.get_relevant_documents(query)

            if not relevant_docs:
                report["status"] = "Consistent (No conflicting context found)"
                report["explanation"] = "No directly relevant or conflicting excerpts found in other FAS via semantic search for comparison."
                report["relevant_other_fas_excerpts"] = "None found."
                return report

            other_fas_context_str = "\n\n---\n\n".join(
                [f"Excerpt from other FAS (Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}"
                 for doc in relevant_docs])
            report["relevant_other_fas_excerpts"] = other_fas_context_str[:1000] + "..." # Truncate for report

            prompt = f"""
            A new suggestion has been proposed for inclusion in AAOIFI Financial Accounting Standard '{fas_name}'.
            Proposed Suggestion Text:
            ---
            {suggestion_text}
            ---

            For context, here are relevant excerpts from *other* AAOIFI Financial Accounting Standards (these are NOT from '{fas_name}'):
            ---
            {other_fas_context_str}
            ---

            Analyze the "Proposed Suggestion Text" for its consistency with the provided "Relevant excerpts from other AAOIFI Financial Accounting Standards".
            Consider:
            1.  Terminological consistency: Does the suggestion use terms in a way that aligns or conflicts with their use in other standards?
            2.  Principle consistency: Does the suggestion introduce principles or treatments that align or conflict with those in other standards?
            3.  Potential for confusion: Could this suggestion, when read alongside other standards, create ambiguity or misunderstanding?

            Provide your analysis as a JSON object with the following keys:
            "status": ("Consistent", "Potential Inconsistency", "Needs Review") - Base this on your findings.
            "explanation": "Your detailed reasoning for the status. Be specific about any terms or principles."
            "conflicting_terms_or_principles": ["List any specific terms or principles that appear inconsistent, or an empty list if none."]

            JSON Output:
            """
            chain = self._create_chain(prompt, output_parser=JsonOutputParser())
            response = self.invoke_chain(chain, {})
            
            report["status"] = response.get("status", "Needs Review")
            report["explanation"] = response.get("explanation", "LLM did not provide detailed explanation for ISCCA.")
            report["conflicting_terms_or_principles"] = response.get("conflicting_terms_or_principles", [])
            if not report["conflicting_terms_or_principles"] and report["status"] == "Consistent": # Refine default
                report["explanation"] = response.get("explanation", "The suggestion appears consistent with the provided excerpts from other standards.")
            return report

        except Exception as e:
            logger.error(f"Error during inter-standard consistency check: {e}", exc_info=True)
            report["status"] = "Error"
            report["explanation"] = f"Error during ISCCA: {str(e)}"
            return report