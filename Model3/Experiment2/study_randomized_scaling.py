"""Unequal-reservoir Model 3: classical randomized-dilation parameter study."""

from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1]))
from library import reservoir_scaling as study


def parse_arguments(arguments=None):
    return study.parse_arguments(arguments, model_name="Model 3 Experiment 2", directory=DIRECTORY, unequal=True)


def main(options):
    return study.main(options)


if __name__ == "__main__":
    main(parse_arguments())
