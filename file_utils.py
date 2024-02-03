import os

def create_dir(save_path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - save_path (str): The path of the directory to be created.
    """
    full_path = os.path.join("run_files", save_path)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
            print(f"Directory created: {full_path}")
        except OSError as e:
            print(f"Error creating directory {full_path}: {e}")
    return full_path