# test_import.py
import os
import sys
from pathlib import Path

def test_imports():
    try:
        from chat.services.llm.llm_manager import EnhancedLLMManager
        print("✓ Successfully imported EnhancedLLMManager")
    except ImportError as e:
        print(f"✗ Import failed: {str(e)}")

if __name__ == "__main__":
    test_imports()