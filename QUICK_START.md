# 🚀 Quick Start Guide

## 1. Environment Configuration

Run this code in `terminal` to install dependencies
```bash
pip install -r requirements.txt
```

Add your API keys to the file [`.env`](/.env)
```bash
DEEPSEEK_API_KEY= your_deepseek_api_key_here
OPENAI_API_KEY= your_openai_api_key_here
```

## 2. Run Experiment Test

Run a single 10-period test to verify everything works:

```bash
# Run following code in terminal
python experiment_runner/test_experiment.py \
  --prompt-type P1 \
  --run-id 1 \
  --num-periods 10
```

**Expected output:**
- Period-by-period price decisions
- Market outcomes (sales, profits, market share)
- `data/runs/P1_run_1/` directory created
- `simulation_log.json` with 10-period data
- `market_history.json` in agent subdirectories

## 3. Run Main Experiment

Run the full experiment (10 runs of 200-period experiments for each prompt type):

```bash
python experiment_runner/main_experiment.py
```

Or start with a smaller experiment for quick review:
```bash
python experiment_runner/main_experiment.py \
  --prompt-types P1 \
  --num-runs 2 \
  --num-periods 50
```

## 4. Results Analysis

Run the interactive jupyternotebook to see the experiment results and data analysis:

```bash
python results_analysis/analysis.ipynb
```