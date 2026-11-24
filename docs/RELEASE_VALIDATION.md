# Publication-release validation — 24 November 2026

- Copied the current local analysis sources without changing their numerical implementations: 45 Python files, two MATLAB helpers, and the existing exploratory notebook. Included the shared package and JSON case catalog absent from the previous remote snapshot.
- All 14 existing regression tests passed with the local Python environment and observation archive (`python -m unittest discover -s . -p 'test_*.py' -v`). The initial sandboxed attempt could not load SciPy's native-library directory; the same suite passed outside that filesystem restriction.
- `python run_analysis.py --check` passed for all 59 workflow-specific preset selections. This verifies exact selected file paths, not independent data quality or scientific reproducibility.
- Parsed all 45 Python source files and checked the README's coverage of all Python and MATLAB files. Checked relative documentation links and image targets.
- Compared the repository PDF byte for byte with the supplied final publication PDF. Rendered and visually inspected the article title page and the four selected figures, retaining captions and source attribution.
- The local environment emits the known SciPy/NumPy supported-version warning and a NumPy binary-compatibility warning. No dependency upgrades were made, no clean environment was validated, and no full figure-regeneration or MHD simulation run is claimed.
- Removed tracked caches, editor metadata, generated large geometry HTML exports, and temporary review artifacts from the new repository tree. Original local files and prior Git history remain intact.

The article's DOI, author list, funding acknowledgment, license, and Zenodo archive identifiers were checked against the supplied final publisher PDF. The external DOI/archive pages were not accessible through the web lookup during preparation; their availability and catalog compatibility were not independently confirmed.
