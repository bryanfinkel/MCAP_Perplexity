# test_structure.py
import os
import sys
from pathlib import Path

def check_structure():
    print("Checking project structure...")
    
    # Define expected directories
    expected_dirs = [
        'chat',
        'chat/services',
        'chat/services/llm',
        'chat/services/search',
        'chat/services/storage',
        'chat/services/embeddings',
        'chat/utils',
        'logs'
    ]
    
    # Define expected files
    expected_files = [
        'chat/__init__.py',
        'chat/services/__init__.py',
        'chat/services/llm/__init__.py',
        'chat/services/llm/llm_manager.py',
        'chat/utils/__init__.py',
        'chat/utils/logging_config.py'
    ]
    
    # Check directories
    print("\nChecking directories:")
    for dir_path in expected_dirs:
        exists = os.path.isdir(dir_path)
        print(f"{dir_path}: {'✓' if exists else '✗'}")
    
    # Check files
    print("\nChecking files:")
    for file_path in expected_files:
        exists = os.path.isfile(file_path)
        print(f"{file_path}: {'✓' if exists else '✗'}")

if __name__ == "__main__":
    check_structure()