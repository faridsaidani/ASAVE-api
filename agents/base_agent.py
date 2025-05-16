# agents/base_agent.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import Runnable
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all AI agents, providing common LLM initialization and chain invocation.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.3, system_message: str = None):
        """
        Initializes the agent with a Google Gemini LLM.

        Args:
            model_name (str): The specific Gemini model to use.
            temperature (float): The sampling temperature for the LLM.
            system_message (str, optional): A system message to prepend to prompts.
        """
        if not os.getenv("GOOGLE_API_KEY"):
            # Let this error propagate; API server / calling code should handle it before agent instantiation.
            logger.error("GOOGLE_API_KEY not set during BaseAgent init.")
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.model_name = model_name
        self.system_message = system_message
        logger.info(f"BaseAgent initialized with LLM: {model_name}, Temperature: {temperature}")

    def _create_prompt_template(self, template_string: str) -> ChatPromptTemplate:
        """Creates a chat prompt template, optionally prepending a system message."""
        if self.system_message:
            messages = [("system", self.system_message), ("human", template_string)]
            return ChatPromptTemplate.from_messages(messages)
        return ChatPromptTemplate.from_template(template_string)

    def _create_chain(self, prompt_template_str: str, output_parser=None) -> Runnable:
        """
        Helper method to create a Langchain chain (LCEL) with a prompt and LLM.

        Args:
            prompt_template_str (str): The string for the prompt template.
            output_parser (Optional): Langchain output parser (e.g., StrOutputParser, JsonOutputParser).
                                     Defaults to StrOutputParser if None.

        Returns:
            Runnable: A Langchain runnable sequence.
        """
        prompt = self._create_prompt_template(prompt_template_str)
        if output_parser is None:
            output_parser = StrOutputParser()
        
        chain = prompt | self.llm | output_parser
        return chain

    def invoke_chain(self, chain: Runnable, input_data: dict) -> any:
        """
        Invokes a Langchain chain with the given input data.

        Args:
            chain (Runnable): The Langchain runnable to invoke.
            input_data (dict): The data to pass to the chain's prompt.

        Returns:
            any: The output from the chain.
        """
        # Log only keys to avoid large data in logs, especially for context strings
        logger.debug(f"Invoking chain with input keys: {list(input_data.keys())}")
        try:
            response = chain.invoke(input_data)
            return response
        except Exception as e:
            logger.error(f"Error invoking chain: {e}", exc_info=True)
            raise