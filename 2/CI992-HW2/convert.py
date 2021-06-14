import json

with open('CI992_HW2.ipynb', 'r') as f:
  notebook = json.load(f)
  
questions = ['Q1.A. Kohonen', 'Q1.B. Kohonen', 'Q2.A. MLP', 'Q2.B. RBF','Q2.C. Compare']
cells = notebook['cells']

for q in questions:
  with open(f"{q}.py", 'w') as f:
    f.write('')
  curq = None
  for cell in cells:
    if cell['cell_type'] == 'markdown' and q in cell['source'][0]:
      curq = q
      continue
    elif cell['cell_type'] == 'markdown' and curq is not None and q not in cell['source'][0] and '# ' in 	cell['source'][0] :
      break
    elif cell['cell_type'] == 'code' and curq is not None:
      with open(f"{q}.py", 'a+') as f:
        f.writelines(cell['source'])
        f.write('\n\n')
