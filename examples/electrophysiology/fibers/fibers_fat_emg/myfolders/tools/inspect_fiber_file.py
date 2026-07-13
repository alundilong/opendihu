#!/usr/bin/env python3

import argparse
import math
import struct
from pathlib import Path


def inspect_fiber_file(filename: str) -> None:
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Fiber file does not exist: {path}")

    with path.open("rb") as f:
        header_raw = f.read(32)

        if len(header_raw) != 32:
            raise RuntimeError("Could not read the 32-byte header.")

        header = struct.unpack("32s", header_raw)[0]
        header_text = header.decode("utf-8", errors="replace").rstrip("\x00")

        header_length_raw = f.read(4)
        if len(header_length_raw) != 4:
            raise RuntimeError("Could not read header length.")

        header_length = struct.unpack("i", header_length_raw)[0]

        if header_length < 8 or header_length % 4 != 0:
            raise RuntimeError(
                f"Unexpected header length: {header_length}"
            )

        n_parameters = header_length // 4 - 1
        parameters = []

        for _ in range(n_parameters):
            raw = f.read(4)
            if len(raw) != 4:
                raise RuntimeError("Unexpected end of header.")
            parameters.append(struct.unpack("i", raw)[0])

    print(f"File:                  {path}")
    print(f"Header:                {header_text}")
    print(f"Header length:         {header_length}")
    print(f"Raw parameters:        {parameters}")

    if len(parameters) < 2:
        raise RuntimeError(
            "The file header does not contain the expected parameters."
        )

    n_fibers_total = parameters[0]
    n_points_per_fiber = parameters[1]

    if "version 2" in header_text.lower():
        if len(parameters) < 4:
            raise RuntimeError(
                "Version-2 file does not contain n_fibers_x/y."
            )

        n_fibers_x = parameters[2]
        n_fibers_y = parameters[3]
    else:
        n_fibers_x = int(round(math.sqrt(n_fibers_total)))
        n_fibers_y = n_fibers_x

    print()
    print(f"n_fibers_total:        {n_fibers_total}")
    print(f"n_fibers_x:            {n_fibers_x}")
    print(f"n_fibers_y:            {n_fibers_y}")
    print(f"n_points_per_fiber:    {n_points_per_fiber}")
    print(
        f"Computed x*y:           "
        f"{n_fibers_x * n_fibers_y}"
    )

    expected = 37 * 37

    print()
    if n_fibers_total == expected:
        print("OK: The file contains 37 x 37 = 1369 fibers.")
    else:
        print(
            f"ERROR: Expected {expected} fibers, "
            f"but the header reports {n_fibers_total}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect an OpenDiHu binary fiber file."
    )
    parser.add_argument("fiber_file", help="OpenDiHu binary fiber file")
    args = parser.parse_args()

    inspect_fiber_file(args.fiber_file)