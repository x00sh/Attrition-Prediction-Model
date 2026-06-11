#!/usr/bin/env python3
"""
Combine .py and .md source files into a Jupyter notebook (.ipynb).

Usage:
  python combine_notebook.py notebooks/modeling.py notebooks/modeling.md
  # → writes notebooks/modeling.ipynb (inferred from .py path)

  python combine_notebook.py notebooks/modeling.py notebooks/modeling.md notebooks/modeling.ipynb
  # → writes notebooks/modeling.ipynb (explicit path)

The .md file must have exactly one more section (delimited by bare ### lines) than the .py file.
The first .md section becomes the notebook intro (no matching code); remaining sections pair with code cells.
"""

import json
import re
import sys
from pathlib import Path
from uuid import uuid4


def split_sections(text: str) -> list[str]:
    """Split text on bare ### delimiters. Return non-empty sections with stripped whitespace."""
    sections = re.split(r'(?m)^###$', text)
    return [s.strip() for s in sections if s.strip()]


def build_notebook(md_sections: list[str], code_sections: list[str]) -> dict:
    """Build a Jupyter notebook from markdown and code sections."""
    if len(md_sections) != len(code_sections) + 1:
        raise ValueError(
            f"Expected len(md_sections) == len(code_sections) + 1, "
            f"got {len(md_sections)} md sections and {len(code_sections)} code sections"
        )

    cells = []

    # Intro markdown cell (no matching code)
    cells.append(_build_markdown_cell(md_sections[0]))

    # Alternating markdown + code pairs
    for i in range(len(code_sections)):
        cells.append(_build_markdown_cell(md_sections[i + 1]))
        cells.append(_build_code_cell(code_sections[i]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def _build_markdown_cell(text: str) -> dict:
    """Build a markdown cell."""
    source_lines = text.split('\n')
    # Every line except the last ends with \n
    source = [line + '\n' for line in source_lines[:-1]]
    if source_lines[-1]:  # Add last line without \n if non-empty
        source.append(source_lines[-1])

    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }


def _build_code_cell(text: str) -> dict:
    """Build a code cell."""
    source_lines = text.split('\n')
    source = [line + '\n' for line in source_lines[:-1]]
    if source_lines[-1]:
        source.append(source_lines[-1])

    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Combine .py and .md files into a Jupyter notebook."
    )
    parser.add_argument("py_file", help="Path to the .py source file")
    parser.add_argument("md_file", help="Path to the .md source file")
    parser.add_argument(
        "ipynb_file",
        nargs="?",
        help="Output .ipynb file (default: inferred from .py path)",
    )
    args = parser.parse_args()

    py_path = Path(args.py_file)
    md_path = Path(args.md_file)

    if not py_path.exists():
        print(f"Error: {py_path} does not exist", file=sys.stderr)
        sys.exit(1)
    if not md_path.exists():
        print(f"Error: {md_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Infer output path if not provided
    if args.ipynb_file:
        ipynb_path = Path(args.ipynb_file)
    else:
        ipynb_path = py_path.with_suffix('.ipynb')

    # Load and split sections
    py_text = py_path.read_text()
    md_text = md_path.read_text()

    py_sections = split_sections(py_text)
    md_sections = split_sections(md_text)

    # Build and write notebook
    try:
        nb = build_notebook(md_sections, py_sections)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    ipynb_path.write_text(json.dumps(nb, indent=1) + '\n')
    print(f"✓ Written {ipynb_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    import argparse
    main()