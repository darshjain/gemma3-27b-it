import subprocess

# Command to install the specified version of the transformers library
subprocess.check_call([ "pip", "install", "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3" ])