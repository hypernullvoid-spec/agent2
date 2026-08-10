"""
Run this FIRST. It needs no API key and no internet.

    python3 test_local.py "/home/spoo/Downloads/movies 2.csv"

It proves steps 1-3 work on your real file before you spend anything.
"""

import sys

import loader


def main(path):
    print(f"Reading {path}")
    tables = loader.read_file(path)
    print(f"Found {len(tables)} table(s): {', '.join(tables)}")

    for name, rows in tables.items():
        raw_size = loader.rough_tokens(loader.to_csv(rows))

        rows = loader.clean(rows)
        clean_size = loader.rough_tokens(loader.to_csv(rows))

        loader.print_weights(rows, name)

        print()
        print(f"  before cleaning : ~{raw_size:>9,d} tokens")
        print(f"  after cleaning  : ~{clean_size:>9,d} tokens", end="")
        print(f"   ({raw_size / max(1, clean_size):.1f}x smaller)")

        # What a light question would actually send
        light = [col for col, pct, _ in loader.column_weights(rows) if pct < 15][:5]
        if light:
            light_size = loader.rough_tokens(loader.to_csv(rows, light))
            print(f"  light columns   : ~{light_size:>9,d} tokens  ({', '.join(light)})")

        print()
        print("  VERDICT:", "fits comfortably" if clean_size < 300_000
              else "still large - the picker will trim it further")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/spoo/Downloads/movies 2.csv")
