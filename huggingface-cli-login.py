import subprocess
import sys


# run command huggingface-cli login
def run_huggingface_login():
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_huggingface_login()
# This script automates the process of logging into Hugging Face CLI.
# It uses the subprocess module to run the 'huggingface-cli login' command.
# Make sure you have Hugging Face CLI installed and accessible in your PATH.
# You can run this script in your terminal to log in.
