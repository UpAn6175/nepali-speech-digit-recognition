"""Copy the per-speaker recordings into one folder per number, the layout the notebooks read.

The raw data is stored per speaker (1_Organized(22)/, 2_Organized(58)/, ...), with
the number in each file name: NEP_S2_D(25)_r1.wav. The notebooks expected
"All Files/<number>/*.wav". This script builds that layout from the file names.

Usage:  python tools/organize_dataset.py "path/to/Digit Recognition" "path/to/All Files"
"""
import re
import shutil
import sys
from pathlib import Path


def main(src, dst):
    src, dst = Path(src), Path(dst)
    count = 0
    for wav in sorted(src.glob("*/*.wav")):
        m = re.search(r"D\((\d+)\)", wav.name)
        if not m:
            print(f"Skipped (no D(n) in name): {wav}")
            continue
        out = dst / m.group(1)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, out / wav.name)
        count += 1
    print(f"Copied {count} files into {len(list(dst.iterdir()))} number folders in {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
