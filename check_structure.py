# check_structure.py
import os

def print_directory_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        # Get the relative path depth
        level = root.replace(startpath, '').count(os.sep)
        # Print the directory with indentation
        print('│   ' * (level) + '├── ' + os.path.basename(root))
        # Print all files in the directory with indentation
        for f in files:
            print('│   ' * (level + 1) + '├── ' + f)

if __name__ == "__main__":
    print("\nProject Structure:")
    print("==================")
    print_directory_structure('chat')