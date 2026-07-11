#!/usr/bin/env python3
"""
Assign motor-unit ownership to fiber streamlines stored in a VTP file.

Each fiber must be stored as one VTK polyline cell. The ownership text file
must contain one integer MU ID for each fiber in the original streamline list.

Example
-------
python assign_mu_ids_to_vtp.py \
    fiber_streamlines.vtp \
    MU_fibre_distribution_37x37_20.txt \
    fiber_streamlines_with_mu.vtp
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pyvista as pv


def load_mu_distribution(filename: str | Path) -> np.ndarray:
    """
    Load MU ownership values from a whitespace- or comma-separated text file.

    Lines may contain comments beginning with '#'.
    """

    filename = Path(filename)

    if not filename.is_file():
        raise FileNotFoundError(
            f'MU distribution file does not exist: "{filename}"'
        )

    cleaned_lines = []

    with filename.open("r", encoding="utf-8") as file:
        for line in file:
            # Remove comments
            line = line.split("#", maxsplit=1)[0]
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Extract signed integers. This supports spaces, tabs, newlines, and commas.
    values = re.findall(r"[-+]?\d+", text)

    if not values:
        raise ValueError(
            f'No integer MU identifiers were found in "{filename}".'
        )

    mu_ids = np.asarray(values, dtype=np.int64)

    return mu_ids


def convert_to_integer_indices(
    values: np.ndarray,
    field_name: str,
) -> np.ndarray:
    """
    Convert a VTK scalar array to integer indices and verify that it contains
    integer-valued data.
    """

    values = np.asarray(values).reshape(-1)

    rounded_values = np.rint(values)

    if not np.allclose(values, rounded_values):
        raise ValueError(
            f'The VTP field "{field_name}" contains non-integer values.'
        )

    return rounded_values.astype(np.int64)


def assign_mu_ids(
    fibers: pv.PolyData,
    mu_distribution: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Determine the MU ID associated with every fiber/polyline cell.

    Assignment priority:
      1. Use zero-based cell field "fiber_id".
      2. Use one-based cell field "fiber_number".
      3. Use direct cell order when cell count equals ownership count.
    """

    n_cells = fibers.n_cells
    n_ownership_values = len(mu_distribution)

    # Preferred method: zero-based original fiber index
    if "fiber_id" in fibers.cell_data:
        fiber_ids = convert_to_integer_indices(
            fibers.cell_data["fiber_id"],
            "fiber_id",
        )

        if len(fiber_ids) != n_cells:
            raise ValueError(
                f'"fiber_id" contains {len(fiber_ids)} values, but the VTP '
                f"contains {n_cells} cells."
            )

        if fiber_ids.size == 0:
            raise ValueError('The "fiber_id" array is empty.')

        if fiber_ids.min() < 0:
            raise ValueError(
                'The "fiber_id" field contains negative indices.'
            )

        if fiber_ids.max() >= n_ownership_values:
            raise ValueError(
                f'The largest "fiber_id" is {fiber_ids.max()}, but the '
                f"ownership file contains only {n_ownership_values} values."
            )

        assigned_mu_ids = mu_distribution[fiber_ids]

        return assigned_mu_ids, "fiber_id"

    # Alternative: one-based fiber number
    if "fiber_number" in fibers.cell_data:
        fiber_numbers = convert_to_integer_indices(
            fibers.cell_data["fiber_number"],
            "fiber_number",
        )

        if len(fiber_numbers) != n_cells:
            raise ValueError(
                f'"fiber_number" contains {len(fiber_numbers)} values, but '
                f"the VTP contains {n_cells} cells."
            )

        if fiber_numbers.size == 0:
            raise ValueError('The "fiber_number" array is empty.')

        if fiber_numbers.min() < 1:
            raise ValueError(
                'The "fiber_number" field must use one-based numbering.'
            )

        if fiber_numbers.max() > n_ownership_values:
            raise ValueError(
                f'The largest "fiber_number" is {fiber_numbers.max()}, but '
                f"the ownership file contains only "
                f"{n_ownership_values} values."
            )

        assigned_mu_ids = mu_distribution[fiber_numbers - 1]

        return assigned_mu_ids, "fiber_number"

    # Fallback: assume cell ordering is identical to text-file ordering
    if n_cells == n_ownership_values:
        return mu_distribution.copy(), "cell order"

    raise ValueError(
        "The MU ownership could not be matched to the VTP fibers.\n"
        f"  Number of VTP cells: {n_cells}\n"
        f"  Number of ownership values: {n_ownership_values}\n"
        '  Available cell-data fields: '
        f"{list(fibers.cell_data.keys())}\n"
        'The VTP should contain either "fiber_id" or "fiber_number", or its '
        "number of cells must equal the number of ownership values."
    )


def add_mu_data(
    input_vtp: str | Path,
    distribution_txt: str | Path,
    output_vtp: str | Path,
) -> None:
    """Load a fiber VTP, assign MU IDs, and save the updated VTP."""

    input_vtp = Path(input_vtp)
    output_vtp = Path(output_vtp)

    if not input_vtp.is_file():
        raise FileNotFoundError(
            f'Input VTP file does not exist: "{input_vtp}"'
        )

    if input_vtp.suffix.lower() != ".vtp":
        raise ValueError(
            f'Input file must use the .vtp extension: "{input_vtp}"'
        )

    if output_vtp.suffix.lower() != ".vtp":
        raise ValueError(
            f'Output file must use the .vtp extension: "{output_vtp}"'
        )

    dataset = pv.read(input_vtp)

    if not isinstance(dataset, pv.PolyData):
        raise TypeError(
            f'"{input_vtp}" is not a VTK PolyData dataset.'
        )

    if dataset.n_cells == 0:
        raise ValueError("The input VTP contains no cells.")

    if dataset.n_lines != dataset.n_cells:
        raise ValueError(
            "The input VTP contains cell types other than lines/polylines.\n"
            f"  Total cells: {dataset.n_cells}\n"
            f"  Line cells:  {dataset.n_lines}\n"
            "This script expects each fiber to be represented by one "
            "polyline cell."
        )

    mu_distribution = load_mu_distribution(distribution_txt)

    output_mesh = dataset.copy(deep=True)

    assigned_mu_ids, matching_method = assign_mu_ids(
        output_mesh,
        mu_distribution,
    )

    # Store the original MU identifiers exactly as they appear in the file.
    output_mesh.cell_data["MU_id"] = assigned_mu_ids.astype(np.int32)

    # Add the number of fibers belonging to the same MU.
    unique_mu_ids, fiber_counts = np.unique(
        assigned_mu_ids,
        return_counts=True,
    )

    count_lookup = {
        int(mu_id): int(count)
        for mu_id, count in zip(unique_mu_ids, fiber_counts)
    }

    output_mesh.cell_data["MU_fiber_count"] = np.asarray(
        [count_lookup[int(mu_id)] for mu_id in assigned_mu_ids],
        dtype=np.int32,
    )

    output_vtp.parent.mkdir(parents=True, exist_ok=True)
    output_mesh.save(output_vtp)

    print("")
    print("MU ownership assignment completed.")
    print(f'Input VTP:              "{input_vtp}"')
    print(f'MU distribution:        "{distribution_txt}"')
    print(f'Output VTP:             "{output_vtp}"')
    print(f"Number of VTP fibers:    {output_mesh.n_cells}")
    print(f"Ownership values loaded: {len(mu_distribution)}")
    print(f"Matching method:         {matching_method}")
    print(f"Number of represented MUs: {len(unique_mu_ids)}")
    print("")
    print("Fiber count for each MU:")

    for mu_id, count in zip(unique_mu_ids, fiber_counts):
        print(f"  MU {int(mu_id):4d}: {int(count):5d} fibers")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign motor-unit IDs from a text file to fiber polylines "
            "stored in a VTP file."
        )
    )

    parser.add_argument(
        "input_vtp",
        help="Input VTP containing fiber streamlines.",
    )

    parser.add_argument(
        "distribution_txt",
        help="Text file containing one MU ID for each original fiber.",
    )

    parser.add_argument(
        "output_vtp",
        help="Output VTP containing the new MU_id cell-data field.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    add_mu_data(
        input_vtp=args.input_vtp,
        distribution_txt=args.distribution_txt,
        output_vtp=args.output_vtp,
    )


if __name__ == "__main__":
    main()