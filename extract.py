import json

notebook_path = r'C:\Users\elango\Downloads\finalminiproject.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('extract_out.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if "pkl" in source or "joblib" in source or "dump" in source or "save" in source or "model" in source.lower():
                out.write(f"--- Cell {i} ---\n")
                out.write(source + "\n")
                out.write("-" * 40 + "\n")
