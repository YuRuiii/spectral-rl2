import os

def rename_subfolders(base_folder):
    for root, dirs, files in os.walk(base_folder):
        for folder_name in dirs:
            if folder_name.startswith("medium"):
                parts = folder_name.split('_')
                if len(parts) == 3:
                    new_folder_name = parts[0]
                    old_path = os.path.join(root, folder_name)
                    new_path = os.path.join(root, new_folder_name)
                    print(f"Renaming {old_path} to {new_path}")
                    os.rename(old_path, new_path)
                    
def print_folder_names(base_folder):
    for root, dirs, files in os.walk(base_folder):
        for folder_name in dirs:
            if folder_name.startswith("medium"):
                print(os.path.join(root, folder_name))

if __name__ == "__main__":
    print_folder_names("log2")