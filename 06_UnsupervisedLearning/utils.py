
import pandas as pd
from pathlib import Path

def convert_to_csv():
    # Caminho da pasta do script
    script_dir = Path(__file__).parent
    xlsx_file = script_dir / 'barrettII_eyes_clustering.xlsx'
    df = pd.read_excel(xlsx_file)

    # Salvamos o CSV na mesma pasta do script
    csv_file = script_dir / 'barrettII_eyes_clustering.csv'
    df.to_csv(csv_file, index=False)