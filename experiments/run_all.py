import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.env_info import write_environment_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    write_environment_info()
    if args.dry_run:
        print("dry-run: environment file written")
        return
    print("run_all: no full runs implemented yet")


if __name__ == "__main__":
    main()
