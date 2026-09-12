"""Command-line entry point for support ticket classification."""

import argparse
import json
from dataclasses import asdict

from ai_engineering_lab.classifier import classify_ticket


def main() -> None:
    """Read one ticket from the command line and print its classification."""
    parser = argparse.ArgumentParser(description="Classify a support ticket.")
    parser.add_argument("ticket_text", help="The support ticket text to classify.")
    arguments = parser.parse_args()

    result = classify_ticket(arguments.ticket_text)
    print(json.dumps(asdict(result)))
