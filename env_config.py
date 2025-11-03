"""
Environment Configuration

Automatically loads API keys from .env file and makes them available globally.
This module should be imported at the top of any file that needs API keys.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find and load .env file from project root
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment variables from {env_file}")
else:
    print(f"⚠ Warning: .env file not found at {env_file}")
    print("  API keys will be loaded from system environment variables")

# Export API keys as module-level variables
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate keys are present
def validate_keys(require_deepseek=True, require_openai=False):
    """
    Validate that required API keys are present.

    Args:
        require_deepseek: Whether DeepSeek key is required
        require_openai: Whether OpenAI key is required

    Raises:
        ValueError: If required keys are missing
    """
    missing = []

    if require_deepseek and not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")

    if require_openai and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing)}\n"
            f"Please add them to the .env file in the project root.\n"
            f"Example .env file:\n"
            f"  DEEPSEEK_API_KEY=sk-your-key-here\n"
            f"  OPENAI_API_KEY=sk-your-key-here"
        )

# Print status on import (can be silenced by setting SILENT_ENV_LOAD=1)
if not os.getenv('SILENT_ENV_LOAD'):
    if DEEPSEEK_API_KEY:
        print(f"✓ DEEPSEEK_API_KEY loaded (length: {len(DEEPSEEK_API_KEY)})")
    else:
        print("✗ DEEPSEEK_API_KEY not found")

    if OPENAI_API_KEY:
        print(f"✓ OPENAI_API_KEY loaded (length: {len(OPENAI_API_KEY)})")
    else:
        print("✗ OPENAI_API_KEY not found")
