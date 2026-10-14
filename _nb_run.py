import subprocess, sys
r = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
     "--execute", "--inplace",
     "--ExecutePreprocessor.timeout=120",
     "--ExecutePreprocessor.allow_errors=True",
     "notebooks/p3_lifestyle.ipynb"],
    capture_output=True, text=True
)
print("exit:", r.returncode)
print(r.stderr[-500:])
