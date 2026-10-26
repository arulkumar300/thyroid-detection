import json

notebook_path = r'C:\Users\elango\Downloads\finalminiproject.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook_outputs.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code' and cell.get('outputs'):
            out_f.write(f"=== CELL {i} OUTPUTS ===\n")
            for out in cell['outputs']:
                text = out.get('text', [])
                if isinstance(text, list):
                    out_f.write("".join(text) + "\n")
                elif isinstance(text, str):
                    out_f.write(text + "\n")
                
                # If it has data like images or tables
                if 'data' in out:
                    for mime_type in out['data']:
                        out_f.write(f"  [MIME Type: {mime_type}]\n")
                        if 'text/plain' in mime_type:
                            data_text = out['data'][mime_type]
                            if isinstance(data_text, list):
                                out_f.write("".join(data_text) + "\n")
                            else:
                                out_f.write(str(data_text) + "\n")
            out_f.write("="*40 + "\n")

print("Saved notebook outputs to notebook_outputs.txt")
