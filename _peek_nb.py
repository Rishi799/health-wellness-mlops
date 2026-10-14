import json, pathlib
nb = json.loads(pathlib.Path("notebooks/p3_lifestyle.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])[:100]
    out_count = len(cell.get("outputs", []))
    print(f"Cell {i:02d} [{cell['cell_type']}] outputs={out_count}: {src[:100].strip()}")
