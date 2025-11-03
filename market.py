"""
Logit Bertrand Market Model

Implements the economic model for duopolistic price competition with differentiated products.

The demand for firm i's product is:
    d_i = β * exp((g - p_i)/μ) / (Σ_j exp((g - p_j)/μ) + 1)

The profit for firm i is:
    π_i = (p_i - c_i) * d_i
"""

import math
from typing import Tuple, Dict
from config.market_config import (
    BETA, PRODUCT_QUALITY, SUBSTITUTABILITY, MARGINAL_COST, NUM_AGENTS
)


class LogitBertrandMarket:
    """
    Implements the Logit Bertrand duopoly market model.
    """

    def __init__(
        self,
        beta: float = BETA,
        quality: float = PRODUCT_QUALITY,
        substitutability: float = SUBSTITUTABILITY,
        marginal_cost: float = MARGINAL_COST
    ):
        """
        Initialize the market model.

        Args:
            beta: Scale parameter for quantity (default: 100)
            quality: Product quality parameter g (default: 2.0)
            substitutability: Product substitutability μ (default: 0.4)
            marginal_cost: Marginal cost c for both firms (default: 1.0)
        """
        self.beta = beta
        self.quality = quality
        self.substitutability = substitutability
        self.marginal_cost = marginal_cost

    def calculate_utility(self, price: float) -> float:
        """
        Calculate the exponential utility term: exp((g - p)/μ)

        Args:
            price: Price of the product

        Returns:
            Exponential utility value
        """
        return math.exp((self.quality - price) / self.substitutability)

    def calculate_demand(self, price_i: float, price_j: float) -> float:
        """
        Calculate demand for firm i given both firms' prices.

        Formula: d_i = β * exp((g - p_i)/μ) / (exp((g - p_i)/μ) + exp((g - p_j)/μ) + 1)

        Args:
            price_i: Price set by firm i
            price_j: Price set by competitor firm j

        Returns:
            Demand (quantity sold) for firm i
        """
        utility_i = self.calculate_utility(price_i)
        utility_j = self.calculate_utility(price_j)
        outside_option = 1.0  # Consumers can choose not to buy (utility = 0, exp(0) = 1)

        denominator = utility_i + utility_j + outside_option
        demand = self.beta * utility_i / denominator

        return demand

    def calculate_profit(self, price: float, demand: float) -> float:
        """
        Calculate profit given price and demand.

        Formula: π = (p - c) * d

        Args:
            price: Price set by the firm
            demand: Demand/quantity sold

        Returns:
            Profit earned
        """
        return (price - self.marginal_cost) * demand

    def calculate_market_share(self, demand_i: float, demand_j: float) -> float:
        """
        Calculate market share for firm i.

        Args:
            demand_i: Demand for firm i
            demand_j: Demand for firm j

        Returns:
            Market share as a percentage (0-100)
        """
        total_demand = demand_i + demand_j
        if total_demand == 0:
            return 50.0  # Equal share if no sales
        return (demand_i / total_demand) * 100

    def simulate_period(self, price_0: float, price_1: float) -> Dict[str, Dict[str, float]]:
        """
        Simulate one period of market interaction given both firms' prices.

        Args:
            price_0: Price set by firm 0
            price_1: Price set by firm 1

        Returns:
            Dictionary containing market outcomes for both firms:
            {
                'firm_0': {'price': float, 'demand': float, 'profit': float, 'market_share': float},
                'firm_1': {'price': float, 'demand': float, 'profit': float, 'market_share': float}
            }
        """
        # Calculate demands
        demand_0 = self.calculate_demand(price_0, price_1)
        demand_1 = self.calculate_demand(price_1, price_0)

        # Calculate profits
        profit_0 = self.calculate_profit(price_0, demand_0)
        profit_1 = self.calculate_profit(price_1, demand_1)

        # Calculate market shares
        market_share_0 = self.calculate_market_share(demand_0, demand_1)
        market_share_1 = self.calculate_market_share(demand_1, demand_0)

        return {
            'firm_0': {
                'price': round(price_0, 2),
                'demand': round(demand_0, 2),
                'profit': round(profit_0, 2),
                'market_share': round(market_share_0, 2)
            },
            'firm_1': {
                'price': round(price_1, 2),
                'demand': round(demand_1, 2),
                'profit': round(profit_1, 2),
                'market_share': round(market_share_1, 2)
            }
        }

    def validate_price(self, price: float, min_price: float, max_price: float) -> Tuple[bool, float]:
        """
        Validate and clip price to acceptable range.

        Args:
            price: Proposed price
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price

        Returns:
            Tuple of (is_valid, clipped_price)
            - is_valid: False if price needed clipping
            - clipped_price: Price clipped to valid range
        """
        if price < min_price:
            return False, min_price
        elif price > max_price:
            return False, max_price
        else:
            return True, price


# Convenience function for easy access
def create_market() -> LogitBertrandMarket:
    """
    Create a market instance with default configuration.

    Returns:
        LogitBertrandMarket instance
    """
    return LogitBertrandMarket()
