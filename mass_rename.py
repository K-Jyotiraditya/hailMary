import os
from pathlib import Path

def mass_rename():
    files = list(Path('.').glob('**/*.py'))
    changed = 0
    for filepath in files:
        if 'venv' in str(filepath):
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'BENCHMARK_INDEX' in content:
                content = content.replace('BENCHMARK_INDEX', 'BENCHMARK_INDEX')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                changed += 1
        except Exception:
            pass
    print(f"Renamed BENCHMARK_INDEX to BENCHMARK_INDEX in {changed} python files.")

if __name__ == '__main__':
    mass_rename()
