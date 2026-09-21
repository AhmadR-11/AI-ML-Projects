#!/usr/bin/env python3
"""
setup_project.py
Automated setup script for MNIST Digit Recognition project structure.
"""

import os

def create_structure(base_path="mnist-digit-recognition"):
    directories = [
        os.path.join(base_path, "data", "raw"),
        os.path.join(base_path, "data", "processed"),
        os.path.join(base_path, "notebooks"),
        os.path.join(base_path, "models"),
        os.path.join(base_path, "outputs", "plots"),
        os.path.join(base_path, "outputs", "reports"),
        os.path.join(base_path, "src"),
    ]

    print(f"Creating project folder structure in: '{base_path}'...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitkeep file so empty directories are tracked by Git
        gitkeep_file = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_file):
            with open(gitkeep_file, "w") as f:
                f.write("")
        print(f"  [CREATED] {directory}")

    # Create requirements.txt if not existing
    req_file = os.path.join(base_path, "requirements.txt")
    if not os.path.exists(req_file):
        req_content = (
            "numpy\n"
            "pandas\n"
            "matplotlib\n"
            "seaborn\n"
            "scikit-learn\n"
            "tensorflow\n"
            "jupyter\n"
            "notebook\n"
            "kaggle\n"
        )
        with open(req_file, "w") as f:
            f.write(req_content)
        print(f"  [CREATED] {req_file}")
    else:
        print(f"  [EXISTS]  {req_file}")

    print("\nProject structure generated successfully!")

if __name__ == "__main__":
    create_structure()
