# agents/text_reformatter_agent.py
from typing import Dict, Any
from .base_agent import BaseAgent # Assuming BaseAgent is in the same directory
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)

class TextReformatterAgent(BaseAgent):
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.1):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            system_message=(
                "You are an expert text processing AI. Your task is to take raw text extracted from a PDF page, "
                "clean it up, identify its structure (paragraphs, headings, lists), and reformat it into clean, "
                "well-structured Markdown. Focus on readability and preserving the original meaning. "
                "Remove redundant line breaks that are artifacts of PDF extraction, but preserve intentional paragraph breaks. "
                "Identify and format headings using Markdown syntax (e.g., #, ##). Format lists correctly. "
                "Ensure code blocks or special text segments are appropriately formatted if identifiable."
            )
        )
        self.agent_type = "AI_TextReformatter"

    def reformat_to_markdown(self, raw_page_text: str, page_number: int = 0) -> Dict[str, Any]:
        if not raw_page_text.strip():
            return {"status": "success", "markdown_content": "", "original_length": 0, "markdown_length": 0, "notes": "Input text was empty."}

        prompt_template = """
        The following text was extracted from a single page of a PDF document (Page {page_number}).
        Please clean up this raw text and reformat it into clear, well-structured Markdown.

        Guidelines:
        1.  Remove artifacts from PDF extraction, such as unnecessary hyphenation at line breaks (if a word is clearly broken across lines but is a single word), and excessive or inconsistent line breaks within paragraphs.
        2.  Merge lines that clearly belong to the same paragraph. Use a single newline for line breaks within a paragraph if appropriate for readability (like poetry or code), but generally aim for continuous paragraph text. Use double newlines to separate distinct paragraphs.
        3.  Identify and format headings using Markdown (e.g., `# Heading 1`, `## Heading 2`).
        4.  Identify and format bulleted or numbered lists correctly using Markdown syntax (`- item` or `1. item`).
        5.  If you detect any tabular data that is poorly formatted in the raw text, try to represent it in a simple Markdown table format if feasible, or clearly delineate it. If too complex, just format the text cleanly.
        6.  Ensure that the core meaning and all essential information from the raw text are preserved.
        7.  Do NOT add any new content or commentary that is not present in the original text. Your role is to clean and reformat.
        8.  If there are any repeated lines or very similar consecutive lines that seem like extraction errors (like the example "carried out by the Islamic banks and financial institutions.\ncarried out by the Islamic banks and financial institutions.(1)"), please consolidate them into a single, correct line.

        Raw Text from Page {page_number}:
        ---
        {raw_text}
        ---

        Cleaned and Reformatted Markdown Output:
        """
        chain = self._create_chain(prompt_template, output_parser=StrOutputParser())
        
        input_data = {
            "raw_text": raw_page_text,
            "page_number": page_number
        }
        agent_method_name = f"{self.agent_type}_reformat_to_markdown"

        try:
            logger.info(f"Invoking {agent_method_name} for page {page_number}, text length {len(raw_page_text)}...")
            markdown_output = self.invoke_chain(chain, input_data)
            logger.info(f"Markdown reformatting complete for page {page_number}.")
            return {
                "status": "success",
                "markdown_content": markdown_output.strip(),
                "original_length": len(raw_page_text),
                "markdown_length": len(markdown_output.strip()),
                "notes": "Successfully reformatted to Markdown."
            }
        except Exception as e:
            logger.error(f"Error in {agent_method_name} for page {page_number}: {e}", exc_info=True)
            return {
                "status": "error",
                "markdown_content": raw_page_text, # Return raw text on error
                "error_message": str(e),
                "notes": "Failed to reformat, returning raw text."
            }