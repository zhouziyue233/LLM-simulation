"""
Main Experiment Runner
Conducts full experiments: 10 runs of 200 periods for each prompt type (P1, P2)
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.env_config import DEEPSEEK_API_KEY
from config.market_config import MIN_PRICE, MAX_PRICE
from simulation_engine.market import LogitBertrandMarket
from simulation_engine.agent import PricingAgent
from simulation_engine.data_manager import DataManager


def run_single_experiment(
    prompt_type: str,
    run_id: int,
    num_periods: int,
    api_key: str
):
    """
    Run a single experiment (one run of N periods).

    Args:
        prompt_type: Either "P1" (defensive) or "P2" (offensive)
        run_id: Unique identifier for this run (1-10)
        num_periods: Number of periods to simulate (default: 200)
        api_key: DeepSeek API key
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {prompt_type} - Run {run_id} - {num_periods} periods")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Initialize components
    market = LogitBertrandMarket()
    data_manager = DataManager(prompt_type=prompt_type, run_id=run_id)

    agents = [
        PricingAgent(agent_id=0, prompt_type=prompt_type, api_key=api_key),
        PricingAgent(agent_id=1, prompt_type=prompt_type, api_key=api_key)
    ]

    # Initialize with random prices
    import random
    random.seed(run_id * 42)  # Reproducible randomness
    prices = [random.uniform(MIN_PRICE, MAX_PRICE) for _ in range(2)]

    print(f"Period 0 (Initial): Agent 0 = ${prices[0]:.2f}, Agent 1 = ${prices[1]:.2f}")

    # Simulate initial period
    results = market.simulate_period(prices[0], prices[1])
    for agent_id in [0, 1]:
        own_firm = f'firm_{agent_id}'
        other_firm = f'firm_{1-agent_id}'

        # Use append_to_history with correct flat format
        data_manager.append_to_history(
            agent_id=agent_id,
            period=0,
            own_price=results[own_firm]['price'],
            own_sales=results[own_firm]['demand'],
            own_profit=results[own_firm]['profit'],
            market_share=results[own_firm]['market_share'],
            competitor_price=results[other_firm]['price']
        )
        data_manager.save_reasoning_process(agent_id, "Initial random pricing.", period=0)

    # Run simulation
    for period in range(1, num_periods + 1):
        if period % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Period {period}/{num_periods} (Elapsed: {elapsed/60:.1f}min)")

        prices = []
        reasonings = {}  # Store reasoning from both agents

        for agent_id in [0, 1]:
            # Load history and reasoning
            history = data_manager.load_market_history(agent_id)
            formatted_history = data_manager.format_market_history_for_prompt(history)
            prev_reasoning = data_manager.load_reasoning_process(agent_id)

            # Get pricing decision (with retry logic)
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    price, reasoning = agents[agent_id].get_pricing_decision(
                        formatted_history,
                        prev_reasoning
                    )
                    prices.append(price)
                    reasonings[agent_id] = reasoning  # Store reasoning for simulation log
                    data_manager.save_reasoning_process(agent_id, reasoning, period=period)
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"  Agent {agent_id}: Attempt {attempt+1} failed, retrying...")
                        time.sleep(2)  # Wait before retry
                    else:
                        raise Exception(f"Agent {agent_id} failed after {max_attempts} attempts: {str(e)}")

        # Simulate market outcomes
        results = market.simulate_period(prices[0], prices[1])

        # Save to history - use append_to_history with correct flat format
        for agent_id in [0, 1]:
            own_firm = f'firm_{agent_id}'
            other_firm = f'firm_{1-agent_id}'

            data_manager.append_to_history(
                agent_id=agent_id,
                period=period,
                own_price=results[own_firm]['price'],
                own_sales=results[own_firm]['demand'],
                own_profit=results[own_firm]['profit'],
                market_share=results[own_firm]['market_share'],
                competitor_price=results[other_firm]['price']
            )

        # Save period log with reasoning from both agents
        data_manager.save_period_results(
            period,
            results,
            reasoning_0=reasonings[0],
            reasoning_1=reasonings[1]
        )

    # Save metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "prompt_type": prompt_type,
        "run_id": run_id,
        "num_periods": num_periods,
        "status": "completed",
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "elapsed_seconds": int(elapsed_time)
    }
    data_manager.save_metadata(metadata)

    print(f"\n{'='*70}")
    print(f"Experiment completed successfully!")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Data saved to: {data_manager.run_dir}")
    print(f"{'='*70}\n")


def run_batch_experiments(
    prompt_types: list,
    num_runs: int = 10,
    num_periods: int = 200,
    api_key: str = None
):
    """
    Run batch experiments for multiple prompt types.

    Args:
        prompt_types: List of prompt types to test (e.g., ['P1', 'P2'])
        num_runs: Number of runs per prompt type (default: 10)
        num_periods: Number of periods per run (default: 200)
        api_key: DeepSeek API key (uses .env if not provided)
    """
    api_key = api_key or DEEPSEEK_API_KEY
    if not api_key:
        raise ValueError(
            "API key not found. Please add DEEPSEEK_API_KEY to .env file."
        )

    print(f"\n{'#'*70}")
    print(f"BATCH EXPERIMENTS")
    print(f"Prompt types: {prompt_types}")
    print(f"Runs per type: {num_runs}")
    print(f"Periods per run: {num_periods}")
    print(f"Total experiments: {len(prompt_types) * num_runs}")
    print(f"{'#'*70}\n")

    batch_start = time.time()
    completed = 0
    failed = 0

    for prompt_type in prompt_types:
        for run_id in range(1, num_runs + 1):
            try:
                run_single_experiment(
                    prompt_type=prompt_type,
                    run_id=run_id,
                    num_periods=num_periods,
                    api_key=api_key
                )
                completed += 1
            except Exception as e:
                print(f"\n❌ FAILED: {prompt_type} Run {run_id}")
                print(f"Error: {str(e)}\n")
                failed += 1

    # Summary
    batch_elapsed = time.time() - batch_start
    print(f"\n{'#'*70}")
    print(f"BATCH EXPERIMENTS COMPLETED")
    print(f"Completed: {completed}/{len(prompt_types) * num_runs}")
    print(f"Failed: {failed}")
    print(f"Total time: {batch_elapsed/3600:.1f} hours")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run main experiments (10 runs × 200 periods)"
    )
    parser.add_argument(
        '--prompt-types',
        nargs='+',
        choices=['P1', 'P2'],
        default=['P1', 'P2'],
        help="Prompt types to test (default: P1 P2)"
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help="Number of runs per prompt type (default: 10)"
    )
    parser.add_argument(
        '--num-periods',
        type=int,
        default=200,
        help="Number of periods per run (default: 200)"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help="DeepSeek API key (or use .env file)"
    )

    # For single run mode
    parser.add_argument(
        '--single-run',
        action='store_true',
        help="Run single experiment instead of batch"
    )
    parser.add_argument(
        '--prompt-type',
        type=str,
        choices=['P1', 'P2'],
        help="Prompt type for single run mode"
    )
    parser.add_argument(
        '--run-id',
        type=int,
        help="Run ID for single run mode"
    )

    args = parser.parse_args()

    try:
        if args.single_run:
            # Single run mode
            if not args.prompt_type or not args.run_id:
                print("Error: --prompt-type and --run-id required for single run mode")
                sys.exit(1)

            api_key = args.api_key or DEEPSEEK_API_KEY
            if not api_key:
                print("Error: API key not found. Please add DEEPSEEK_API_KEY to .env file.")
                sys.exit(1)

            run_single_experiment(
                prompt_type=args.prompt_type,
                run_id=args.run_id,
                num_periods=args.num_periods,
                api_key=api_key
            )
        else:
            # Batch mode
            run_batch_experiments(
                prompt_types=args.prompt_types,
                num_runs=args.num_runs,
                num_periods=args.num_periods,
                api_key=args.api_key
            )

    except Exception as e:
        print(f"\nExperiment failed: {str(e)}")
        sys.exit(1)
