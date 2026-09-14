# About

The rest of the site shows you what the library does. These four pages are for
deciding whether to believe it, and what happens to your code when it changes.

| | |
|---|---|
| **[Validation](validation.md)** | How any of this is known to be correct: an independent reference sharing no code, cross-library parity against PennyLane, property-based fuzzing, cross-backend equivalence, and executable documentation — ordered by how much each one can prove |
| **[API stability](stability.md)** | What is promised, what is internal, and what a version number means here |
| **[Changelog](changelog.md)** | Every release, and — more usefully — the failures that prompted each fix, written out rather than summarised |
| **[Releasing](releasing.md)** | The tag-to-PyPI procedure, for whoever cuts the next one |

Validation is the one to read first if you are deciding whether to adopt the library
at all. It leads with the check that can prove the most and ends with the ones that
can prove the least, and it records where the project has had to correct itself —
twice on its own benchmark numbers, both times in its own favour, both caught by
someone re-running the comparison rather than trusting the table.
