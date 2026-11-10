"""The README's Python examples run as a reader runs them.

A ``python`` block runs unless its first line is ``# doc: skip``. As a
script, the release's installed-package gate runs the blocks against the
built wheel; it imports only the standard library:

    python tests/test_readme.py README.md
"""

import pathlib
import re
import sys

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def blocks(readme: pathlib.Path) -> list[str]:
    found = BLOCK.findall(readme.read_text())
    return [code for code in found if not code.lstrip().startswith("# doc: skip")]


def run(code: str, readme: pathlib.Path) -> None:
    exec(compile(code, str(readme), "exec"), {"__name__": "__readme__"})


def test_the_readme_has_examples():
    """Guards the test below: a broken fence pattern finds no block."""
    assert blocks(README)


def test_the_readme_examples_run():
    for code in blocks(README):
        run(code, README)


if __name__ == "__main__":
    for path in sys.argv[1:] or [str(README)]:
        for code in blocks(pathlib.Path(path)):
            run(code, pathlib.Path(path))
    print("README examples ran.")
