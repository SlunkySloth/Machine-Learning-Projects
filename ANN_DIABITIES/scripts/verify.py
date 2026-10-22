"""Run the meaningful regression checks and save their actual output."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    with (ROOT / "reports/verification.txt").open("w") as log:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(stream=Tee(sys.stdout, log), verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
