"""The version is written in three places and must agree: gtweb.__version__ (gtweb/__init__.py), pyproject.toml
[project] version, and CITATION.cff.  A release that bumps one and not the others fails here."""
import re
from pathlib import Path

import gtweb

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version():
    text = (ROOT / "pyproject.toml").read_text()
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[)", text, re.S | re.M).group(1)
    return re.search(r'^version\s*=\s*"([^"]+)"', project, re.M).group(1)


def _citation_version():
    return re.search(r"^version:\s*(\S+)\s*$", (ROOT / "CITATION.cff").read_text(), re.M).group(1)


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", gtweb.__version__), gtweb.__version__


def test_init_pyproject_and_citation_agree():
    assert gtweb.__version__ == _pyproject_version() == _citation_version()


def test_changelog_has_a_section_for_this_version():
    assert re.search(rf"^## {re.escape(gtweb.__version__)} \(", (ROOT / "CHANGELOG.md").read_text(), re.M)
