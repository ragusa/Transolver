"""Package-level help for Heat2D command entry points."""

HELP = """Heat2D command modules:

Dataset and checking:
  python -m fom_generation.heat2d.cli.generate_dataset --help
  python -m fom_generation.heat2d.cli.generate_pilot_dataset --help
  python -m fom_generation.heat2d.cli.generate_balanced_dataset --help
  python -m fom_generation.heat2d.cli.check_dataset --help

Validation:
  python -m fom_generation.heat2d.validation.manufactured --help
  python -m fom_generation.heat2d.validation.materials --help
  python -m fom_generation.heat2d.validation.geometry_randomization --help
  python -m fom_generation.heat2d.validation.geometry_stress --help

Tests:
  python -m pytest tests/heat2d -q
"""


def main():
    print(HELP)


if __name__ == "__main__":
    main()
