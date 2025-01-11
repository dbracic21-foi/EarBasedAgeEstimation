import os
import subprocess
import sys

def install_requirements():
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError:

def run_script(script_name):
   
    print(f"\n🚀 Starting {script_name}...\n")
    result = subprocess.run(["python", f"src/{script_name}"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":

    install_requirements()
    run_script("src/preprocess.py")
    run_script("src/split_data.py")
    run_script("src/train.py")
