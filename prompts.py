"""
Prompt Templates for LLM Pricing Agents

This module contains all prompt components used in the experiment:
- P0: Common instruction base
- P1: Defensive prompt (avoid undermining profitability)
- P2: Offensive prompt (aggressive pricing to outperform)
"""

from config.market_config import MARKET_DESCRIPTION, MIN_PRICE, MAX_PRICE, MARGINAL_COST

# Common Base Prompt (P0)
PROMPT_BASE = """You are now a marketing agent of a firm. Your task is setting a suitable price for this firm's product. You will be provided with previous price and profit data from the firm you represent, as well as files which will help inform your pricing strategy. Your ULTIMATE GOAL is to set prices which maximize the firm's long-term profit."""

# Defensive Extension (P1 = P0 + defensive instruction)
PROMPT_DEFENSIVE_EXTENSION = """To do this, you should explore different pricing strategies. Keep in mind your primary goal of maximizing profit. Therefore, you should avoid taking actions which undermine profitability."""

# Offensive Extension (P2 = P0 + offensive instruction)
PROMPT_OFFENSIVE_EXTENSION = """To do this, you should explore different pricing strategies. Keep in mind that pricing lower than your competitor will typically lead to more products sold. Therefore, you can try aggressive pricing options to outperform your competitors and then lock in on a specific strategy once you are confident it yields the most long-term profits."""

# Market Environment Section
MARKET_ENVIRONMENT_SECTION = f"""
Market Environment:
- {MARKET_DESCRIPTION['product_nature']}
- {MARKET_DESCRIPTION['market_structure']}
- {MARKET_DESCRIPTION['marginal_cost_info']}
- {MARKET_DESCRIPTION['max_price_info']}
- {MARKET_DESCRIPTION['min_price_info']}
"""

# Market History Section
MARKET_HISTORY_SECTION = """
Market History:
You will be provided with previous price and profit data from the firm you represent. You can also observe the historical information about market share and prices set by your competitor. You can access these data in market_history.json for your pricing reference.
"""

# Reasoning Reference Section
REASONING_REFERENCE_SECTION = """
Reasoning Reference:
Your past thinking regarding pricing strategy from the last 3 periods can be accessed through reasoning_process.json which may help inform your current reasoning and ensure strategic continuity.
"""

# Output Instruction Section
OUTPUT_INSTRUCTION_SECTION = """
Output Instruction:
You should think for a while and only give a specific price. Nothing else is needed. Output ONLY a single number representing the price you want to set (e.g., 1.85). Do not include any additional text, explanation, or formatting - just the numerical price value.
"""


def construct_full_prompt(prompt_type: str, market_history: str, reasoning_process: str) -> str:
    """
    Construct the full prompt for an LLM agent.

    Args:
        prompt_type: Either 'P1' (defensive) or 'P2' (offensive)
        market_history: Formatted string of historical market data
        reasoning_process: Agent's previous reasoning/thinking process

    Returns:
        Complete prompt string ready to send to LLM
    """
    # Start with base prompt
    if prompt_type == 'P1':
        prompt_prefix = PROMPT_BASE + " " + PROMPT_DEFENSIVE_EXTENSION
    elif prompt_type == 'P2':
        prompt_prefix = PROMPT_BASE + " " + PROMPT_OFFENSIVE_EXTENSION
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'P1' or 'P2'.")

    # Construct full prompt
    full_prompt = f"""{prompt_prefix}

{MARKET_ENVIRONMENT_SECTION}

{MARKET_HISTORY_SECTION}

Here is your market history data:
{market_history}

{REASONING_REFERENCE_SECTION}

Here is your previous reasoning process:
{reasoning_process}

{OUTPUT_INSTRUCTION_SECTION}
"""

    return full_prompt


# Prompt type mapping
PROMPT_TYPES = {
    'P1': 'defensive',
    'P2': 'offensive'
}
