#!/usr/bin/env python3
"""CLI wrapper for dataprep.features.aggregation.ticker_batch_runner"""
import os, sys, runpy

repo_root = os.path.dirname(os.path.abspath(__file__))  # cli/
repo_root = os.path.dirname(repo_root)                  # repo/
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

if __name__ == "__main__":
    runpy.run_module("dataprep.features.aggregation.ticker_batch_runner", run_name="__main__")
