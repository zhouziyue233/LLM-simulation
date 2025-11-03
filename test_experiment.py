"""
Test Experiment Runner
Quick test with small number of periods for debugging and validation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.env_config import DEEPSEEK_API_KEY
from config.market_config import MIN_PRICE, MAX_PRICE
from simulation_engine.market import LogitBertrandMarket
from simulation_engine.agent import PricingAgent
from simulation_engine.data_manager import DataManager


def run_test_experiment(
    prompt_type: str = "P1",
    num_periods: int = 10,
    run_id: int = 0,
    api_key: str = None
):
    """
    Run a quick test experiment with 10 periods.

    Args:
        prompt_type: Either "P1" (defensive) or "P2" (offensive)
        num_periods: Number of periods to simulate (default: 10)
        run_id: Run identifier (default: 0)
        api_key: DeepSeek API key (uses .env if not provided)
    """
    # Use API key from .env if not provided
    api_key = api_key or DEEPSEEK_API_KEY
    if not api_key:
        raise ValueError(
            "API key not found. Please add DEEPSEEK_API_KEY to .env file "
            "or pass api_key argument."
        )

    print(f"\n{'='*60}")
    print(f"TEST EXPERIMENT: {prompt_type} - Run {run_id} - {num_periods} periods")
    print(f"{'='*60}\n")

    # Initialize components
    market = LogitBertrandMarket()
    data_manager = DataManager(prompt_type=prompt_type, run_id=run_id)

    agents = [
        PricingAgent(agent_id=0, prompt_type=prompt_type, api_key=api_key),
        PricingAgent(agent_id=1, prompt_type=prompt_type, api_key=api_key)
    ]

    # Initialize with random prices
    import random
    prices = [random.uniform(MIN_PRICE, MAX_PRICE) for _ in range(2)]

    print(f"Period 0 (Initial): Agent 0 = ${prices[0]:.2f}, Agent 1 = ${prices[1]:.2f}")

    # Simulate initial period
    results = market.simulate_period(prices[0], prices[1])
    # Convert nested results to flat format for saving
    flat_results = {
        'price_0': results['firm_0']['price'],
        'price_1': results['firm_1']['price'],
        'demand_0': results['firm_0']['demand'],
        'demand_1': results['firm_1']['demand'],
        'profit_0': results['firm_0']['profit'],
        'profit_1': results['firm_1']['profit'],
        'market_share_0': results['firm_0']['market_share'],
        'market_share_1': results['firm_1']['market_share']
    }
    for agent_id in [0, 1]:
        competitor_id = 1 - agent_id
        data_manager.append_to_history(
            agent_id=agent_id,
            period=0,
            own_price=results[f'firm_{agent_id}']['price'],
            own_sales=results[f'firm_{agent_id}']['demand'],
            own_profit=results[f'firm_{agent_id}']['profit'],
            market_share=results[f'firm_{agent_id}']['market_share'],
            competitor_price=results[f'firm_{competitor_id}']['price']
        )
        data_manager.save_reasoning_process(agent_id, "Initial random pricing.", period=0)

    # Run simulation
    for period in range(1, num_periods + 1):
        print(f"\n--- Period {period} ---")

        prices = []
        reasonings = {}  # Store reasoning from both agents

        for agent_id in [0, 1]:
            # Load history and reasoning
            history = data_manager.load_market_history(agent_id)
            # Filter out old format entries and keep only valid entries
            filtered_history = [entry for entry in history if 'period' in entry]
            formatted_history = data_manager.format_market_history_for_prompt(filtered_history, agent_id)
            prev_reasoning = data_manager.load_reasoning_process(agent_id)

            # Get pricing decision
            print(f"Agent {agent_id}: Making pricing decision...")
            price, reasoning = agents[agent_id].get_pricing_decision(
                formatted_history,
                prev_reasoning
            )
            prices.append(price)
            reasonings[agent_id] = reasoning  # Store reasoning for simulation log

            print(f"Agent {agent_id}: Price = ${price:.2f}")

            # Save reasoning to agent's file
            data_manager.save_reasoning_process(agent_id, reasoning, period=period)

        # Simulate market outcomes
        results = market.simulate_period(prices[0], prices[1])

        # Convert nested results to flat format for display and saving
        flat_results = {
            'price_0': results['firm_0']['price'],
            'price_1': results['firm_1']['price'],
            'demand_0': results['firm_0']['demand'],
            'demand_1': results['firm_1']['demand'],
            'profit_0': results['firm_0']['profit'],
            'profit_1': results['firm_1']['profit'],
            'market_share_0': results['firm_0']['market_share'],
            'market_share_1': results['firm_1']['market_share']
        }

        # Display results
        print(f"\nMarket Results:")
        print(f"  Agent 0: Demand={flat_results['demand_0']:.1f}, Profit=${flat_results['profit_0']:.2f}, Share={flat_results['market_share_0']:.1f}%")
        print(f"  Agent 1: Demand={flat_results['demand_1']:.1f}, Profit=${flat_results['profit_1']:.2f}, Share={flat_results['market_share_1']:.1f}%")

        # Save to history for each agent
        for agent_id in [0, 1]:
            competitor_id = 1 - agent_id
            data_manager.append_to_history(
                agent_id=agent_id,
                period=period,
                own_price=results[f'firm_{agent_id}']['price'],
                own_sales=results[f'firm_{agent_id}']['demand'],
                own_profit=results[f'firm_{agent_id}']['profit'],
                market_share=results[f'firm_{agent_id}']['market_share'],
                competitor_price=results[f'firm_{competitor_id}']['price']
            )

        # Save period log with reasoning from both agents
        data_manager.save_period_results(
            period,
            flat_results,
            reasoning_0=reasonings[0],
            reasoning_1=reasonings[1]
        )

    # Save metadata
    metadata = {
        "prompt_type": prompt_type,
        "run_id": run_id,
        "num_periods": num_periods,
        "status": "completed"
    }
    data_manager.save_metadata(metadata)

    print(f"\n{'='*60}")
    print(f"Test experiment completed successfully!")
    print(f"Data saved to: {data_manager.run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a quick test experiment")
    parser.add_argument(
        '--prompt-type',
        type=str,
        choices=['P1', 'P2'],
        default='P1',
        help="Prompt type to use (P1=defensive, P2=offensive)"
    )
    parser.add_argument(
        '--num-periods',
        type=int,
        default=10,
        help="Number of periods to simulate (default: 10)"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help="DeepSeek API key (or use .env file)"
    )
    parser.add_argument(
        '--run-id',
        type=int,
        default=0,
        help="Run identifier (default: 0)"
    )

    args = parser.parse_args()

    try:
        run_test_experiment(
            prompt_type=args.prompt_type,
            num_periods=args.num_periods,
            run_id=args.run_id,
            api_key=args.api_key
        )
    except Exception as e:
        print(f"\nTest experiment failed: {str(e)}")
        sys.exit(1)
