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
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.1):
        super().__init__(model_name=model_name, temperature=temperature,
                         system_message="You are an expert Shari'ah scholar and AAOIFI standards auditor. Your task is to critically evaluate text for compliance and consistency based on provided rules and context.")

    def _load_explicit_shariah_rules(self, rules_file_path: str) -> List[Dict[str, Any]]:
        """Loads Shariah rules from a JSON or JSONL file."""
        try:
            if not os.path.exists(rules_file_path):
                logger.warning(f"Shariah rules file not found: {rules_file_path}")
                return []
                
            if rules_file_path.lower().endswith('.jsonl'):
                # Use the JSONL loader for .jsonl files
                return self._load_jsonl_file(rules_file_path)
            else:
                # Use standard JSON loader for .json files
                with open(rules_file_path, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                    logger.info(f"Successfully loaded rules from JSON file: {rules_file_path}")
                    return rules if isinstance(rules, list) else []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from Shari'ah rules file: {rules_file_path}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error loading Shari'ah rules from {rules_file_path}: {e}", exc_info=True)
            return []

    def _load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a JSONL file (multiple JSON objects, one per line) into a list of dictionaries.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries, each representing a JSON object from a line in the file
        """
        rules = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('//'):  # Skip comments and empty lines
                        try:
                            rule = json.loads(line)
                            rules.append(rule)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing line in {file_path}: {line[:50]}... - {e}")
            logger.info(f"Successfully loaded {len(rules)} rules from JSONL file: {file_path}")
            return rules
        except Exception as e:
            logger.error(f"Error loading JSONL file {file_path}: {e}")
            return []

    def _load_all_shariah_rules_for_prompt(self, shariah_rules_explicit_path: str, mined_shariah_rules_path: Optional[str] = None) -> str:
        """
        Loads all explicit and mined Shari'ah rules and formats them into a single string
        suitable for inclusion in an LLM prompt.
        """
        all_rules_text_parts = []
        rules_loaded_count = 0

        # Load explicit rules
        if shariah_rules_explicit_path and os.path.exists(shariah_rules_explicit_path):
            try:
                with open(shariah_rules_explicit_path, 'r', encoding='utf-8') as f:
                    explicit_rules = json.load(f)
                if isinstance(explicit_rules, list):
                    for rule in explicit_rules:
                        all_rules_text_parts.append(
                            f"Rule ID: {rule.get('rule_id', 'UNKNOWN_ID')}\n"
                            f"Description: {rule.get('description', 'No description.')}\n"
                            f"Principle Keywords: {', '.join(rule.get('principle_keywords', []))}\n"
                            # We won't use the validation_query_template directly in this batched approach's prompt
                            # but including description and keywords is vital.
                        )
                        rules_loaded_count +=1
                    logger.info(f"Loaded {len(explicit_rules)} explicit rules for batch validation prompt.")
            except Exception as e:
                logger.error(f"Error loading explicit Shari'ah rules from {shariah_rules_explicit_path}: {e}")
        
        # Load mined rules
        if mined_shariah_rules_path and os.path.exists(mined_shariah_rules_path):
            try:
                with open(mined_shariah_rules_path, 'r', encoding='utf-8') as f:
                    mined_rules = json.load(f)
                if isinstance(mined_rules, list):
                    for rule in mined_rules:
                         all_rules_text_parts.append(
                            f"Rule ID: {rule.get('rule_id', 'UNKNOWN_MINED_ID')}\n"
                            f"Description: {rule.get('description', 'No description.')}\n"
                            f"Standard Reference: {rule.get('standard_ref', 'N/A')}\n"
                            f"Principle Keywords: {', '.join(rule.get('principle_keywords', []))}\n"
                        )
                         rules_loaded_count += 1
                    logger.info(f"Loaded {len(mined_rules)} mined rules for batch validation prompt.")
            except Exception as e:
                logger.error(f"Error loading mined Shari'ah rules from {mined_shariah_rules_path}: {e}")
        
        if not all_rules_text_parts:
            return "No Shari'ah rules were loaded for reference."
            
        logger.info(f"Total {rules_loaded_count} rules prepared for batch validation prompt.")
        return "\n---\n".join(all_rules_text_parts)

    def validate_shariah_compliance_batched(self, # New method name
                                            proposed_suggestion_object: Dict[str, Any],
                                            shariah_rules_explicit_path: str,
                                            ss_vector_store: Optional[Chroma], # Still useful for broader semantic context
                                            mined_shariah_rules_path: Optional[str] = None,
                                            contract_type: Optional[str] = "General", # Context for semantic search
                                            k_semantic: int = 3) -> Dict[str, Any]:
        """
        Validates a proposed suggestion for Shari'ah compliance against a whole set of rules
        in a single (or fewer) LLM call(s), and performs semantic validation.
        """
        suggestion_text = proposed_suggestion_object.get("proposed_text", "")
        aisga_shariah_notes = proposed_suggestion_object.get("shariah_notes", "N/A") # Notes from the suggestion agent

        # Prepare the final report structure
        report = {
            "overall_status": "Needs Expert Review", # Default until assessed
            "summary_explanation": "Initial assessment pending. Analyzed against a rule set.",
            "explicit_rule_batch_assessment": { # For the batched rule check
                "status_from_llm": "Not Performed", # e.g., "No Violations Found" or "Violations Identified"
                "identified_issues": [], # List of {"rule_id": "...", "concern": "...", "severity": "Potential/Clear Violation"}
                "llm_reasoning": "N/A"
            },
            "semantic_validation_against_ss": { # For the RAG against Shari'ah Standards
                "status": "Not Performed", # e.g., "Aligned", "Potential Misalignment", "No Relevant Context Found"
                "notes": "N/A",
                "relevant_ss_excerpts_summary": "N/A"
            },
            "final_notes_for_expert": ""
        }

        if not suggestion_text:
            report["summary_explanation"] = "No proposed text provided for validation."
            report["overall_status"] = "Error: No Text"; return report

        # 1. Batch Validation against Explicit/Mined Rules
        all_rules_str = self._load_all_shariah_rules_for_prompt(shariah_rules_explicit_path, mined_shariah_rules_path)
        
        if "No Shari'ah rules were loaded" in all_rules_str:
            report["explicit_rule_batch_assessment"]["status_from_llm"] = "Skipped - No Rules"
            report["explicit_rule_batch_assessment"]["llm_reasoning"] = all_rules_str
        else:
            logger.info(f"SCVA Batch: Validating suggestion text (len {len(suggestion_text)}) against rule set (len {len(all_rules_str)}).")
            # Ensure the total prompt length is manageable for the LLM.
            # If all_rules_str is too long, you might need to chunk it or use a more advanced model,
            # or implement a pre-filtering of rules based on contract_type or keywords from suggestion_text.

            batch_prompt_template = """
            You are an expert Shari'ah auditor.
            A "Proposed Text" for a financial document or contract clause is provided below.
            You are also provided with a "Set of Shari'ah Rules".
            Your task is to meticulously review the "Proposed Text" against EACH rule in the "Set of Shari'ah Rules".

            For EACH rule that the "Proposed Text" potentially violates or raises a concern about, identify:
            1. The `Rule ID` of the conflicting rule.
            2. A brief `Concern` explaining how the "Proposed Text" conflicts with or relates to this specific rule.
            3. A `Severity` assessment for the concern (e.g., "Clear Violation", "Potential Violation", "Needs Clarification", "Minor Concern").

            If the "Proposed Text" appears to comply with all provided rules, state that clearly.

            Proposed Text:
            ---
            {suggestion_text}
            ---

            Set of Shari'ah Rules:
            ---
            {all_rules_text}
            ---

            Original Shari'ah Notes for this "Proposed Text" (for context, from the agent that generated it):
            ---
            {aisga_shariah_notes}
            ---
            
            Provide your output as a JSON object with the following structure:
            {{
                "overall_assessment_of_rules": "No Violations Found" OR "Violations Identified" OR "Needs Detailed Review",
                "identified_issues": [ 
                    {{ "rule_id": "...", "concern": "...", "severity": "..." }},
                    // ... more issues if any
                ],
                "summary_reasoning_for_assessment": "Your overall reasoning based on the rule checks."
            }}
            If no issues are found, the "identified_issues" array should be empty.
            """
            # Using a model good with large context and JSON output
            # Consider "gemini-1.5-pro-latest" if the combined text is large.
            # Adjust self.model_name if this agent was initialized with Flash.
            # It might be better to pass the model instance to use for this specific task.
            chain = self._create_chain(batch_prompt_template, output_parser=JsonOutputParser())
            try:
                llm_batch_response = self.invoke_chain(chain, {
                    "suggestion_text": suggestion_text,
                    "all_rules_text": all_rules_str,
                    "aisga_shariah_notes": aisga_shariah_notes
                })
                report["explicit_rule_batch_assessment"]["status_from_llm"] = llm_batch_response.get("overall_assessment_of_rules", "LLM Response Error")
                report["explicit_rule_batch_assessment"]["identified_issues"] = llm_batch_response.get("identified_issues", [])
                report["explicit_rule_batch_assessment"]["llm_reasoning"] = llm_batch_response.get("summary_reasoning_for_assessment", "N/A")
            except Exception as e_llm_batch:
                logger.error(f"LLM batch rule validation failed: {e_llm_batch}", exc_info=True)
                report["explicit_rule_batch_assessment"]["status_from_llm"] = "LLM Error"
                report["explicit_rule_batch_assessment"]["llm_reasoning"] = f"LLM call failed: {str(e_llm_batch)}"
                report["explicit_rule_batch_assessment"]["identified_issues"] = [{"rule_id": "N/A", "concern": "LLM processing error during batch validation.", "severity": "Error"}]


        # 2. Semantic Validation against SS Vector Store (Broader Context)
        # This remains largely the same as your original semantic validation, providing a different kind of check.
        if ss_vector_store:
            logger.info("SCVA Batch: Performing semantic validation against SS vector store...")
            try:
                # Construct a query for semantic search
                semantic_query = f"Shari'ah alignment and principles for the following text proposed for a {contract_type} context: '{suggestion_text}'. Initial notes: '{aisga_shariah_notes}'"
                retriever = ss_vector_store.as_retriever(search_kwargs={"k": k_semantic})
                relevant_ss_docs = retriever.get_relevant_documents(semantic_query)

                if relevant_ss_docs:
                    ss_context_str = "\n\n---\n\n".join([f"SS Excerpt (Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" for doc in relevant_ss_docs])
                    report["semantic_validation_against_ss"]["relevant_ss_excerpts_summary"] = f"Retrieved {len(relevant_ss_docs)} excerpts. First one: {relevant_ss_docs[0].page_content[:200]}..." if relevant_ss_docs else "None"
                    
                    semantic_eval_prompt = f"""
                    Proposed Text:
                    ---
                    {suggestion_text}
                    ---
                    Original Shari'ah Notes for this "Proposed Text" (for context, from the agent that generated it):
                    ---
                    {aisga_shariah_notes}
                    ---
                    Relevant Excerpts from Shari'ah Standards (SS) for general context:
                    ---
                    {ss_context_str}
                    ---
                    Based ONLY on the "Relevant Excerpts from Shari'ah Standards", evaluate the "Proposed Text".
                    Does it appear generally aligned or misaligned with the principles in these excerpts?
                    Provide a brief "evaluation_status" ("Aligned", "Potentially Misaligned", "Neutral/Unclear based on excerpts", "No Relevant Context Found")
                    and a "concise_explanation". Output as a JSON object.
                    """
                    semantic_chain = self._create_chain(semantic_eval_prompt, output_parser=JsonOutputParser())
                    semantic_eval_response = self.invoke_chain(semantic_chain, {})
                    report["semantic_validation_against_ss"]["status"] = semantic_eval_response.get("evaluation_status", "LLM Error")
                    report["semantic_validation_against_ss"]["notes"] = semantic_eval_response.get("concise_explanation", "N/A")
                else:
                    report["semantic_validation_against_ss"]["status"] = "No Relevant Context Found"
                    report["semantic_validation_against_ss"]["notes"] = "No relevant Shari'ah Standard excerpts found via semantic search for this suggestion."
            except Exception as e_semantic:
                logger.error(f"Semantic validation failed: {e_semantic}", exc_info=True)
                report["semantic_validation_against_ss"]["status"] = "Error"
                report["semantic_validation_against_ss"]["notes"] = f"Semantic validation processing error: {str(e_semantic)}"
        else:
            report["semantic_validation_against_ss"]["status"] = "Skipped - No SS Vector Store"
            report["semantic_validation_against_ss"]["notes"] = "Shari'ah Standards vector store was not available for semantic validation."


        # 3. Determine Overall Status and Final Notes (Simplified for this example)
        # This logic could also be a separate LLM call for a more nuanced summary.
        final_issues = report["explicit_rule_batch_assessment"]["identified_issues"]
        if any(issue.get("severity", "").lower() == "clear violation" for issue in final_issues):
            report["overall_status"] = "Non-Compliant"
            report["summary_explanation"] = "Clear violation(s) identified against the provided Shari'ah rule set."
        elif any("violation" in issue.get("severity", "").lower() for issue in final_issues): # Potential violation
            report["overall_status"] = "Needs Expert Review"
            report["summary_explanation"] = "Potential violation(s) or concerns identified against the Shari'ah rule set. Expert review required."
        elif report["explicit_rule_batch_assessment"]["status_from_llm"] == "No Violations Found":
            if "Potentially Misaligned" in report["semantic_validation_against_ss"]["status"]:
                report["overall_status"] = "Needs Expert Review"
                report["summary_explanation"] = "No direct rule violations found, but semantic check against SS excerpts suggests potential misalignment or areas needing review."
            elif report["semantic_validation_against_ss"]["status"] == "Error" or report["semantic_validation_against_ss"]["status"] == "LLM Error":
                report["overall_status"] = "Needs Expert Review"
                report["summary_explanation"] = "No direct rule violations found, but semantic check encountered an error. Expert review recommended."
            else: # Rules look okay, semantic check is Aligned or Neutral
                report["overall_status"] = "Compliant (Initial Automated Check)"
                report["summary_explanation"] = "Initial automated checks against the rule set and semantic context suggest compliance. Final determination requires expert review."
        else: # Default if explicit rule check was inconclusive or had other issues
            report["overall_status"] = "Needs Expert Review"
            report["summary_explanation"] = f"Automated rule set review status: {report['explicit_rule_batch_assessment']['status_from_llm']}. Semantic SS check: {report['semantic_validation_against_ss']['status']}. Expert review strongly recommended."

        report["final_notes_for_expert"] = f"Suggestion: '{suggestion_text[:100]}...'. Rule set assessment: {report['explicit_rule_batch_assessment']['status_from_llm']}. Semantic SS context: {report['semantic_validation_against_ss']['status']}. Key issues from rules: {len(final_issues)} found. Please review details."
        
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

    def review_entire_contract(self,
                               contract_text: str,
                               contract_type: str, # e.g., "Mudarabah", "Salam"
                               shariah_rules_explicit_path: str,
                               mined_shariah_rules_path: Optional[str] = None,
                               ss_vector_store: Optional[Chroma] = None, # For broader SS principles
                               k_semantic_ss: int = 5
                              ) -> Dict[str, Any]:
        """
        Reviews an entire contract text against Shari'ah rules and principles.
        Aims to provide structured feedback identifying issues and suggesting improvements.
        """
        logger.info(f"Reviewing entire contract (length {len(contract_text)}) of type '{contract_type}'.")

        report = {
            "overall_contract_assessment": "Needs Detailed Review",
            "contract_summary_by_ai": "AI has not yet summarized the contract.",
            "identified_clauses_with_issues": [], # List of findings
            "general_recommendations": [],
            "shariah_alignment_notes": "",
            "debug_info": {
                "contract_text_snippet": contract_text[:500] + "..."
            }
        }

        # 1. Load all Shari'ah rules for the prompt (same helper as before)
        all_rules_str = self._load_all_shariah_rules_for_prompt(
            shariah_rules_explicit_path, mined_shariah_rules_path
        )
        if "No Shari'ah rules were loaded" in all_rules_str:
            report["overall_contract_assessment"] = "Cannot Assess - No Rules Loaded"
            report["shariah_alignment_notes"] = "No Shari'ah rules were available for assessment."
            return report

        # 2. Retrieve broader SS principles using RAG (optional but good)
        ss_principles_context = "No specific Shari'ah Standard excerpts retrieved for general principles."
        if ss_vector_store:
            try:
                query_for_ss_principles = f"General Shari'ah principles and common pitfalls for a {contract_type} agreement. Overall context: {contract_text[:1000]}" # Query with snippet
                retriever = ss_vector_store.as_retriever(search_kwargs={"k": k_semantic_ss})
                relevant_ss_docs = retriever.get_relevant_documents(query_for_ss_principles)
                if relevant_ss_docs:
                    ss_principles_context = "\n\n---\n\n".join([doc.page_content for doc in relevant_ss_docs])
                    logger.info(f"Retrieved {len(relevant_ss_docs)} SS excerpts for general principles.")
            except Exception as e_ss_rag:
                logger.error(f"Error retrieving SS principles for contract review: {e_ss_rag}")
                ss_principles_context = "Error retrieving Shari'ah Standard excerpts."
        
        # 3. Construct the main LLM prompt
        # This prompt asks the LLM to act as a reviewer, identify issues clause by clause,
        # and suggest improvements.
        # The LLM needs to be good at identifying clause boundaries itself if not pre-segmented.

        # Make sure total prompt length is considered. If contract_text + all_rules_str is too large,
        # this approach will fail or be very slow/expensive.
        # For very large contracts, a pre-segmentation step for `contract_text` would be needed,
        # then iterate this review process over segments.
        # For now, assuming contract_text is manageable.

        prompt_template = """
        You are an expert Shari'ah compliance reviewer specializing in Islamic finance contracts.
        You will be given the full text of a proposed "{contract_type}" contract and a set of Shari'ah rules and principles.

        Your tasks are:
        1.  Read the entire "Proposed Contract Text" carefully.
        2.  Identify key clauses or sections within the contract.
        3.  For each clause/section, evaluate its compliance against the provided "Set of Shari'ah Rules" and general "Shari'ah Standard Principles".
        4.  Provide a structured list of your findings. For each finding:
            a.  Quote the "original_clause_text" (or a significant identifying snippet) from the contract that your finding pertains to.
            b.  Describe the "issue_or_concern" (e.g., non-compliance, ambiguity, missing information, potential for Gharar/Riba).
            c.  Reference the specific "shariah_rule_ids" (from the provided rule set) that are relevant to this issue, if applicable.
            d.  Suggest a "recommended_action_or_modification" to address the issue and improve compliance or clarity. This could be a rephrased clause or an additional point.
            e.  Indicate a "severity" for the issue (e.g., "High - Clear Non-Compliance", "Medium - Potential Risk/Ambiguity", "Low - Suggestion for Enhancement").
        5.  Provide an "overall_assessment" of the contract's Shari'ah compliance based on your findings.
        6.  Optionally, list any "general_recommendations" for the contract as a whole.

        Proposed Contract Text (Type: {contract_type}):
        ---
        {full_contract_text}
        ---

        Set of Shari'ah Rules for Reference:
        ---
        {all_shariah_rules}
        ---

        General Shari'ah Standard Principles (for broader context):
        ---
        {ss_principles}
        ---

        Please provide your output as a single JSON object with the following structure:
        {{
            "overall_assessment": "e.g., Needs Significant Revision / Mostly Compliant with Minor Issues / Appears Compliant",
            "contract_summary_by_ai": "A brief AI-generated summary of the contract's purpose and key elements.",
            "identified_clauses_with_issues": [
                {{
                    "original_clause_text_snippet": "The relevant snippet from the contract...",
                    "issue_or_concern": "Description of the issue...",
                    "relevant_shariah_rule_ids": ["RULE_ID_1", "RULE_ID_2"],
                    "recommended_action_or_modification": "Suggested change or new text...",
                    "severity": "High / Medium / Low"
                }}
                // ... more findings if any
            ],
            "general_recommendations": [
                "General advice or points to consider for this type of contract..."
            ],
            "overall_shariah_alignment_notes": "Your detailed reasoning for the overall assessment."
        }}
        If no significant issues are found in a generally well-drafted contract, the "identified_clauses_with_issues" array can be empty or contain only items with "Low" severity for enhancement.
        """

        # Use a powerful model for this complex task, e.g., gemini-1.5-pro-latest
        # Ensure this ValidationAgent instance's LLM is suitable.
        if "flash" in self.model_name.lower() and (len(contract_text) + len(all_rules_str) + len(ss_principles_context)) > 70000 : # Rough estimate for Flash token limit for safety
            logger.warning("Input text for whole contract review might be too large for Flash model. Consider Gemini Pro.")
            report["overall_contract_assessment"] = "Error - Input Too Large for Model"
            report["overall_shariah_alignment_notes"] = "The combined contract text and rules are too long for the configured LLM. Analysis cannot proceed."
            return report


        chain = self._create_chain(prompt_template, output_parser=JsonOutputParser())
        try:
            llm_response = self.invoke_chain(chain, {
                "contract_type": contract_type,
                "full_contract_text": contract_text,
                "all_shariah_rules": all_rules_str,
                "ss_principles": ss_principles_context
            })

            report["overall_contract_assessment"] = llm_response.get("overall_assessment", "Needs Detailed Review (LLM Error)")
            report["contract_summary_by_ai"] = llm_response.get("contract_summary_by_ai", "AI did not provide a summary.")
            report["identified_clauses_with_issues"] = llm_response.get("identified_clauses_with_issues", [])
            report["general_recommendations"] = llm_response.get("general_recommendations", [])
            report["shariah_alignment_notes"] = llm_response.get("overall_shariah_alignment_notes", "LLM did not provide detailed alignment notes.")
            
            logger.info(f"Whole contract review completed. Overall Assessment: {report['overall_contract_assessment']}")

        except Exception as e_llm:
            logger.error(f"LLM call for whole contract review failed: {e_llm}", exc_info=True)
            report["overall_contract_assessment"] = "Error During AI Review"
            report["shariah_alignment_notes"] = f"An error occurred during the AI review process: {str(e_llm)}"
            report["identified_clauses_with_issues"] = [{"original_clause_text_snippet": "N/A", "issue_or_concern": "LLM processing error.", "severity": "Error"}]

        return report