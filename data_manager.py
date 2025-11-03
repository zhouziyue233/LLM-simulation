"""
Data Manager for Simulation

Handles all file I/O operations including:
- Reading/writing market history (last 50 periods)
- Reading/writing reasoning process
- Logging simulation data for analysis
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path
from config.market_config import HISTORY_LENGTH, REASONING_HISTORY_LENGTH


class DataManager:
    """
    Manages data persistence for the simulation.
    """

    def __init__(self, prompt_type: str, run_id: int, base_dir: str = "data"):
        """
        Initialize data manager with run-specific directory.

        Args:
            prompt_type: Prompt type (P1 or P2)
            run_id: Run identifier
            base_dir: Base data directory (default: "data")
        """
        project_root = Path(__file__).parent.parent
        self.run_dir = project_root / base_dir / f"{prompt_type}_run_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each agent
        self.agent_dirs = {}
        for agent_id in [0, 1]:
            agent_dir = self.run_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            self.agent_dirs[agent_id] = agent_dir

    def get_market_history_path(self, agent_id: int) -> Path:
        """Get path to market history file for an agent."""
        return self.agent_dirs[agent_id] / "market_history.json"

    def get_reasoning_process_path(self, agent_id: int) -> Path:
        """Get path to reasoning process file for an agent."""
        return self.agent_dirs[agent_id] / "reasoning_process.json"

    def get_simulation_log_path(self) -> Path:
        """Get path to full simulation log file."""
        return self.run_dir / "simulation_log.json"

    def save_market_history(self, agent_id: int, history: List[Dict]) -> None:
        """
        Save market history for an agent (keeping only last HISTORY_LENGTH periods).

        Args:
            agent_id: Agent identifier (0 or 1)
            history: List of historical period data
        """
        # Keep only the last HISTORY_LENGTH periods
        trimmed_history = history[-HISTORY_LENGTH:] if len(history) > HISTORY_LENGTH else history

        file_path = self.get_market_history_path(agent_id)
        with open(file_path, 'w') as f:
            json.dump(trimmed_history, f, indent=2)

    def load_market_history(self, agent_id: int) -> List[Dict]:
        """
        Load market history for an agent.

        Args:
            agent_id: Agent identifier (0 or 1)

        Returns:
            List of historical period data (empty list if file doesn't exist)
        """
        file_path = self.get_market_history_path(agent_id)
        if not file_path.exists():
            return []

        with open(file_path, 'r') as f:
            return json.load(f)

    def format_market_history_for_prompt(self, history: List[Dict], agent_id: int = 0) -> str:
        """
        Format market history into a readable string for the LLM prompt.

        Args:
            history: List of historical period data
            agent_id: Agent ID for formatting (used if old format detected)

        Returns:
            Formatted string representation
        """
        if not history:
            return "No historical data available yet. This is the beginning of the market."

        formatted = "Period | Your Price | Your Sales | Your Profit | Market Share | Competitor Price\n"
        formatted += "-" * 90 + "\n"

        for entry in history:
            if 'period' in entry:
                formatted += (
                    f"{entry['period']:6d} | "
                    f"${entry['own_price']:9.2f} | "
                    f"{entry['own_sales']:10.2f} | "
                    f"${entry['own_profit']:11.2f} | "
                    f"{entry['market_share']:11.2f}% | "
                    f"${entry['competitor_price']:15.2f}\n"
                )

        return formatted if formatted.count('\n') > 2 else "No historical data available yet. This is the beginning of the market."

    def save_reasoning_process(self, agent_id: int, reasoning: str, period: int = None) -> None:
        """
        Save the agent's reasoning process from the current period.
        Maintains a rolling history of the last REASONING_HISTORY_LENGTH periods.

        Args:
            agent_id: Agent identifier (0 or 1)
            reasoning: The reasoning/thinking process text
            period: Period number (optional, for tracking)
        """
        file_path = self.get_reasoning_process_path(agent_id)

        # Load existing reasoning history
        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
                # Handle both old single-entry format and new list format
                if isinstance(existing_data, dict) and 'reasoning' in existing_data:
                    # Old format: convert to list
                    reasoning_history = [existing_data]
                elif isinstance(existing_data, list):
                    # New format: use as is
                    reasoning_history = existing_data
                else:
                    reasoning_history = []
        else:
            reasoning_history = []

        # Append new reasoning
        new_entry = {
            "period": period,
            "reasoning": reasoning
        }
        reasoning_history.append(new_entry)

        # Keep only the last REASONING_HISTORY_LENGTH entries
        reasoning_history = reasoning_history[-REASONING_HISTORY_LENGTH:]

        # Save updated history
        with open(file_path, 'w') as f:
            json.dump(reasoning_history, f, indent=2)

    def load_reasoning_process(self, agent_id: int) -> str:
        """
        Load the agent's previous reasoning process(es).
        Returns formatted text of the last REASONING_HISTORY_LENGTH periods.

        Args:
            agent_id: Agent identifier (0 or 1)

        Returns:
            Formatted previous reasoning text (message if no history exists)
        """
        file_path = self.get_reasoning_process_path(agent_id)
        if not file_path.exists():
            return "No previous reasoning available. This is your first decision."

        with open(file_path, 'r') as f:
            data = json.load(f)

            # Handle both old single-entry format and new list format
            if isinstance(data, dict) and 'reasoning' in data:
                # Old format: single entry
                period = data.get('period', 'unknown')
                reasoning = data.get('reasoning', '')
                return f"[Period {period}]\n{reasoning}"
            elif isinstance(data, list):
                # New format: multiple entries
                if not data:
                    return "No previous reasoning available. This is your first decision."

                # Format multiple periods
                formatted_parts = []
                for entry in data:
                    period = entry.get('period', 'unknown')
                    reasoning = entry.get('reasoning', '')
                    formatted_parts.append(f"[Period {period}]\n{reasoning}")

                return "\n\n" + "="*80 + "\n\n".join(formatted_parts)
            else:
                return "No previous reasoning available. This is your first decision."

    def append_to_history(
        self,
        agent_id: int,
        period: int,
        own_price: float,
        own_sales: float,
        own_profit: float,
        market_share: float,
        competitor_price: float
    ) -> None:
        """
        Append a new period's data to the agent's market history.

        Args:
            agent_id: Agent identifier (0 or 1)
            period: Period number
            own_price: Price set by this agent
            own_sales: Sales/demand for this agent
            own_profit: Profit earned by this agent
            market_share: Market share percentage
            competitor_price: Price set by competitor
        """
        history = self.load_market_history(agent_id)

        new_entry = {
            "period": period,
            "own_price": round(own_price, 2),
            "own_sales": round(own_sales, 2),
            "own_profit": round(own_profit, 2),
            "market_share": round(market_share, 2),
            "competitor_price": round(competitor_price, 2)
        }

        history.append(new_entry)
        self.save_market_history(agent_id, history)

    def save_period_results(self, period: int, results: Dict, reasoning_0: str = None, reasoning_1: str = None) -> None:
        """
        Save period results to simulation log.

        Args:
            period: Period number
            results: Dictionary containing market outcomes
            reasoning_0: Agent 0's reasoning process (optional)
            reasoning_1: Agent 1's reasoning process (optional)
        """
        log_path = self.get_simulation_log_path()

        # Load existing log or create new
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []

        # Add new period data
        period_data = {"period": period, **results}

        # Add reasoning if provided
        if reasoning_0 is not None:
            period_data["reasoning_0"] = reasoning_0
        if reasoning_1 is not None:
            period_data["reasoning_1"] = reasoning_1

        log_data.append(period_data)

        # Save updated log
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def log_simulation_period(self, period: int, results: Dict) -> None:
        """Deprecated: Use save_period_results instead."""
        self.save_period_results(period, results)

    def get_full_simulation_log(self) -> List[Dict]:
        """
        Retrieve the complete simulation log.

        Returns:
            List of all period results
        """
        log_path = self.get_simulation_log_path()
        if not log_path.exists():
            return []

        with open(log_path, 'r') as f:
            return json.load(f)

    def save_metadata(self, metadata: Dict) -> None:
        """
        Save metadata about the simulation run.

        Args:
            metadata: Dictionary containing run metadata (prompt_type, run_id, etc.)
        """
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> Optional[Dict]:
        """
        Load metadata about the simulation run.

        Returns:
            Metadata dictionary or None if file doesn't exist
        """
        metadata_path = self.base_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)
