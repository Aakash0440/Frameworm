
"""Check for missing docstrings and generate report"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class DocstringChecker(ast.NodeVisitor):
    """Check for missing docstrings in Python files"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.missing: List[Tuple[str, int]] = []
    
    def visit_ClassDef(self, node):
        if not ast.get_docstring(node):
            self.missing.append((f"Class {node.name}", node.lineno))
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        if not ast.get_docstring(node):
            # Skip private functions and __init__ without docstring is ok
            if not node.name.startswith('_') or node.name == '__init__':
                if node.name != '__init__':
                    self.missing.append((f"Function {node.name}", node.lineno))
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)


def check_file(filepath: Path) -> List[Tuple[str, int]]:
    """Check single file for missing docstrings"""
    try:
        content = filepath.read_text()
        tree = ast.parse(content)
        checker = DocstringChecker(str(filepath))
        checker.visit(tree)
        return checker.missing
    except:
        return []


def main():
    """Check all Python files in frameworm/"""
    
    print("Checking for missing docstrings...\n")
    
    all_missing = {}
    total_missing = 0
    
    for py_file in Path('frameworm').rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        
        missing = check_file(py_file)
        if missing:
            all_missing[str(py_file)] = missing
            total_missing += len(missing)
    
    # Print report
    if all_missing:
        print(f"Found {total_missing} missing docstrings:\n")
        
        for filepath, missing in sorted(all_missing.items()):
            print(f"{filepath}:")
            for name, lineno in missing:
                print(f"  Line {lineno}: {name}")
            print()
    else:
        print("✓ All public functions and classes have docstrings!")
    
    # Return exit code
    sys.exit(1 if total_missing > 0 else 0)


if __name__ == '__main__':
    main()
