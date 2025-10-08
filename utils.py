import pandas as pd
from pathlib import Path
import sys


def convert_to_csv(xlsx_path):
    xlsx_file = Path(xlsx_path)
    df = pd.read_excel(xlsx_file)

    # Save CSV in the same folder as the input file
    csv_file = xlsx_file.with_suffix(".csv")
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python utils.py <input_xlsx_path>")
        sys.exit(1)
    convert_to_csv(sys.argv[1])
