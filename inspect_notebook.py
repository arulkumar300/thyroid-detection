import json

notebook_path = r'C:\Users\elango\Downloads\finalminiproject.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook_details.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        out.write(f"=== CELL {i} ({cell_type.upper()}) ===\n")
        source = "".join(cell.get('source', []))
        out.write(source + "\n")
        out.write("\n" + "="*80 + "\n\n")

print("Saved notebook details to notebook_details.txt")
