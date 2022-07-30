from pathlib import Path
src_dir = Path('_source')
for file in src_dir.iterdir():
    print('Processed RST file:', file)
    with open(file, 'r', ) as f:
        lines = f.read()

    junk_strs = ['Submodules\n----------', 'Subpackages\n-----------']

    for junk in junk_strs:
        lines = lines.replace(junk, '')

    with open(file, 'w') as f:
        f.write(lines)
