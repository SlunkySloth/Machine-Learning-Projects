"""Train and evaluate the full experiment without installing the project package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from diabetes.train import main

if __name__ == "__main__":
    main()
