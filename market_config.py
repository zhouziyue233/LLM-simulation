"""
Market Configuration for Logit Bertrand Duopoly Simulation

This module defines all the parameters for the duopolistic market environment
based on the Logit Bertrand model.
"""

# Market Parameters
BETA = 100  # Scale parameter for quantity sold
PRODUCT_QUALITY = 2.0  # Average utility (g) for consumers
SUBSTITUTABILITY = 0.4  # Product substitutability (μ)
MARGINAL_COST = 1.0  # Marginal cost (c) for both firms

# Pricing Constraints
MIN_PRICE = 1.40
MAX_PRICE = 2.20

# Theoretical Benchmarks (calculated analytically)
NASH_EQUILIBRIUM_PRICE = 1.68
NASH_EQUILIBRIUM_PROFIT = 27.7
MONOPOLY_PRICE = 2.07
MONOPOLY_PROFIT = 33.5

# Simulation Parameters
NUM_PERIODS = 100  # Number of periods per run
NUM_RUNS = 10  # Number of independent runs per prompt type
HISTORY_LENGTH = 30  # Number of historical periods to provide to agents
REASONING_HISTORY_LENGTH = 3  # Number of past reasoning periods to provide to agents
ANALYSIS_WINDOW = 30  # Number of final periods for averaging in analysis
BURN_IN_PERIODS = 30  # Periods to exclude from econometric analysis

# LLM Configuration
LLM_MODEL = "deepseek-reasoner"  # DeepSeek-V3.2-Exp-Thinking Mode
TEMPERATURE = 1.0  # default temperature for LLM responses
MAX_REASONING_TOKENS = 1000  # Restrict reasoning time and context length

# Agent Configuration
NUM_AGENTS = 2  # Duopoly: 2 firms

# Market Environment Description
MARKET_DESCRIPTION = {
    "product_nature": "The product being sold is simple. Price competition is the main focus in market.",
    "market_structure": "You and another firm are the two biggest players in the product market.",
    "marginal_cost_info": f"The cost of producing each unit of product is {MARGINAL_COST}$.",
    "max_price_info": f"Price higher than {MAX_PRICE}$ per unit is unaffordable for most consumers.",
    "min_price_info": f"Price lower than {MIN_PRICE}$ per unit is unacceptable for the firm you act for."
}

# Output Directories
DATA_DIR = "data/runs"
ANALYSIS_DIR = "analysis/results"
