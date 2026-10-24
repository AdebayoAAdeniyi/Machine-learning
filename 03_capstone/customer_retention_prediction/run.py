#!/usr/bin/env python3
"""Pointer to the retention runner, which needs a dataset you provide.

The customer tables behind this study are not redistributable, so there is
no dataset in this repository for this script to use. The pipeline itself
lives in `scripts/run_customer_retention.py` and takes a prepared numeric
CSV, so it can be read, tested and applied to your own data.
"""
import sys

MESSAGE = """\
This study needs a customer table, which is not bundled with the repository.

Run the pipeline on your own prepared numeric CSV instead:

    python scripts/run_customer_retention.py --csv your_table.csv --target churned
    python scripts/run_customer_retention.py --csv your_table.csv --target churned --smote

See 03_capstone/customer_retention_prediction/README.md for the feature
construction, the split and SMOTE ordering, and how to read the metrics.
"""

if __name__ == '__main__':
    sys.stderr.write(MESSAGE)
    raise SystemExit(1)
