"""
LLM Agent Wrapper for Pricing Decisions

Handles interaction with DeepSeek API and response parsing.
"""

import re
import time
from typing import Tuple, Optional
import openai
from config.market_config import (
    LLM_MODEL, TEMPERATURE, MAX_REASONING_TOKENS, MIN_PRICE, MAX_PRICE
)
from config.prompts import construct_full_prompt


class PricingAgent:
    """
    Wrapper for LLM-based pricing agent.
    """

    def __init__(
        self,
        agent_id: int,
        prompt_type: str,
        api_key: str,
        api_base: Optional[str] = None
    ):
        """
        Initialize the pricing agent.

        Args:
            agent_id: Unique identifier for this agent (0 or 1)
            prompt_type: Type of prompt to use ('P1' or 'P2')
            api_key: API key for DeepSeek
            api_base: Optional custom API base URL
        """
        self.agent_id = agent_id
        self.prompt_type = prompt_type

        # Configure OpenAI client for DeepSeek API
        # DeepSeek API is compatible with OpenAI's API format
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base if api_base else "https://api.deepseek.com"
        )

    def get_pricing_decision(
        self,
        market_history: str,
        reasoning_process: str,
        max_retries: int = 3
    ) -> Tuple[float, str]:
        """
        Get a pricing decision from the LLM agent.

        Args:
            market_history: Formatted market history string
            reasoning_process: Previous reasoning from the agent
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (price, reasoning_text)
            - price: The price decision
            - reasoning_text: The agent's reasoning process

        Raises:
            ValueError: If unable to parse valid price after retries
        """
        # Construct the full prompt
        full_prompt = construct_full_prompt(
            self.prompt_type,
            market_history,
            reasoning_process
        )

        for attempt in range(max_retries):
            try:
                # Call DeepSeek API
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_REASONING_TOKENS
                )

                # Extract response content
                response_content = response.choices[0].message.content

                # Extract reasoning if available (DeepSeek reasoning models provide thinking process)
                reasoning_text = ""
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning_text = response.choices[0].message.reasoning_content

                # Handle empty content (reasoning model may put everything in reasoning_content)
                if not response_content or response_content.strip() == "":
                    if reasoning_text:
                        # Use reasoning_content as fallback
                        response_content = reasoning_text
                    else:
                        print(f"Agent {self.agent_id}: WARNING - Empty response received")
                        print(f"  Finish reason: {response.choices[0].finish_reason}")
                        raise ValueError("Empty response from API")

                # If reasoning_text is empty, use content as fallback
                if not reasoning_text:
                    reasoning_text = response_content

                # Parse price from response (try content first, then reasoning)
                try:
                    price = self._parse_price(response_content)
                except ValueError:
                    # If parsing content fails and we have reasoning, try parsing reasoning
                    if reasoning_text and reasoning_text != response_content:
                        price = self._parse_price(reasoning_text)
                    else:
                        raise

                # Validate price
                if MIN_PRICE <= price <= MAX_PRICE:
                    return price, reasoning_text
                else:
                    # Clip to valid range
                    price = max(MIN_PRICE, min(MAX_PRICE, price))
                    print(f"Agent {self.agent_id}: Price clipped to valid range: {price}")
                    return price, reasoning_text

            except Exception as e:
                print(f"Agent {self.agent_id}: Attempt {attempt + 1} failed: {str(e)}")
                print(f"  Error type: {type(e).__name__}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ValueError(f"Failed to get valid pricing decision after {max_retries} attempts")

    def _parse_price(self, response_text: str) -> float:
        """
        Parse price from LLM response text.

        Args:
            response_text: Raw text response from LLM

        Returns:
            Extracted price value

        Raises:
            ValueError: If no valid price can be extracted
        """
        # Try to find a number in the response
        # Look for patterns like: 1.85, $1.85, 1.85$, "1.85", etc.

        # Remove common currency symbols and whitespace
        cleaned_text = response_text.strip().replace('$', '').replace('€', '').replace('£', '')

        # Try multiple regex patterns
        patterns = [
            r'^(\d+\.?\d*)$',  # Just a number (e.g., "1.85")
            r'(\d+\.\d+)',     # Decimal number (e.g., "The price is 1.85")
            r'(\d+)',          # Integer (e.g., "2")
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                try:
                    price = float(match.group(1))
                    return price
                except ValueError:
                    continue

        # If no valid price found, try to extract any number from the text
        numbers = re.findall(r'\d+\.?\d*', response_text)
        if numbers:
            # Take the first number that falls within reasonable range
            for num_str in numbers:
                try:
                    price = float(num_str)
                    if 0 < price < 100:  # Sanity check
                        return price
                except ValueError:
                    continue

        raise ValueError(f"Could not parse price from response: {response_text}")

    def make_decision(
        self,
        market_history: str,
        reasoning_process: str
    ) -> Tuple[float, str]:
        """
        High-level method to make a pricing decision.

        Args:
            market_history: Formatted market history string
            reasoning_process: Previous reasoning from the agent

        Returns:
            Tuple of (price, reasoning_text)
        """
        print(f"Agent {self.agent_id} ({self.prompt_type}): Making pricing decision...")

        price, reasoning = self.get_pricing_decision(
            market_history,
            reasoning_process
        )

        print(f"Agent {self.agent_id}: Decided on price ${price:.2f}")

        return price, reasoning


def create_agent(agent_id: int, prompt_type: str, api_key: str) -> PricingAgent:
    """
    Factory function to create a pricing agent.

    Args:
        agent_id: Unique identifier for this agent (0 or 1)
        prompt_type: Type of prompt to use ('P1' or 'P2')
        api_key: API key for DeepSeek

    Returns:
        PricingAgent instance
    """
    return PricingAgent(agent_id, prompt_type, api_key)
