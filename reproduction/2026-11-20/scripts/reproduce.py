from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
(root / 'source/tables').mkdir(parents=True, exist_ok=True)
for script in ('analyze_p2.py', 'make_p1_tables.py', 'make_p2_tables.py',
               'make_p3_tables.py', 'check_selection.py', 'make_validation_macros.py',
               'fig1_method.py', 'fig_p1.py', 'fig_p2.py', 'fig_p3.py'):
    subprocess.run([sys.executable, str(root/'scripts'/script)], cwd=root, check=True)
