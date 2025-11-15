## Reporting
Here, we have helper functions and code that we can use to automatically generate reports for in LaTeX.

Usage: simply modify the script to include the listed results to show. Otherwise, leave blank to evaluate all outputs at `results/`, then run:
```bash
python reporting/results2table.py --json_path 'path/to/eval_results.json'
```