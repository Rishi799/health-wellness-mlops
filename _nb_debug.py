import subprocess, sys
r = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
     "--execute", "--inplace", "--ExecutePreprocessor.timeout=120",
     "notebooks/p3_lifestyle.ipynb"],
    capture_output=True, text=True
)
print("STDOUT:", r.stdout[-3000:])
print("STDERR:", r.stderr[-3000:])
print("exit:", r.returncode)
