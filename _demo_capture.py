import subprocess, sys, pathlib
r = subprocess.run(
    [sys.executable, "scripts/validation_demo.py", "--cycle", "2017-2018"],
    capture_output=True, text=True
)
out = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr and "FAIL" in r.stderr else "")
pathlib.Path("reports/demo/demo_output.txt").write_text(out, encoding="utf-8")
print(out)
print(f"\nexit code = {r.returncode}")
