# Paper tables

Numbering follows the current main text and SI Appendix. CSV files retain analysis precision and may include diagnostic columns or additional rows. LaTeX snippets select and round the displayed values, using the paper’s table layouts. Table 1 uses largest-remainder rounding so each three-outcome group totals 1.000.

Rebuild with `python analysis/reproduce_all.py`, or export already computed results with `python analysis/paper_tables.py`. The two descriptive tables use versioned templates in `analysis/table_templates/`. Snippets require the manuscript’s macros and packages.
