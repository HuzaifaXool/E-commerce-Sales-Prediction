import subprocess
import sys
import os

def run_script(script_name):
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"Successfully executed {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")

scripts = ["src/data/data_loader.py", "src/data/feature_engineering.py", "src/vizualization/data_visualization.py",
"src/training/train_pipeline.py"]

for script in scripts:
    if not os.path.exists(script):
        print(f"Warning: {script} not found!")
    else:
        run_script(script)
