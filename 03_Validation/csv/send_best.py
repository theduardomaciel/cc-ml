import argparse
import json
from pathlib import Path
import pandas as pd
import requests

OUT_DIR = Path(__file__).parent / "out" / "all_models"
SERVER_URL = "https://aydanomachado.com/mlclass/03_Validation.php"
RESPONSE_FILE = Path(__file__).parent / "response.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Envio de previsões já salvas do melhor modelo"
    )
    parser.add_argument(
        "--best-model", type=str, help="Nome do modelo cujas previsões serão enviadas"
    )
    parser.add_argument(
        "--dev-key",
        type=str,
        default="COLOCAR_SUA_KEY_AQUI",
        help="Chave do desenvolvedor",
    )
    args = parser.parse_args()

    pred_csv_path = OUT_DIR / f"{args.best_model}_predictions.csv"
    pred_json_path = OUT_DIR / f"{args.best_model}_predictions.json"

    if not pred_csv_path.exists() or not pred_json_path.exists():
        print(
            f"Arquivo de previsões do modelo {args.best_model} não encontrado em {OUT_DIR}."
        )
        return

    # Lê previsões salvas
    y_pred = pd.read_csv(pred_csv_path).iloc[:, 0].tolist()

    if not args.dev_key or args.dev_key == "COLOCAR_SUA_KEY_AQUI":
        print(
            "DEV_KEY não informado. Use --dev-key ou defina variável de ambiente DEV_KEY."
        )
        return

    # Monta payload
    payload = {
        "dev_key": args.dev_key,
        "predictions": json.dumps(y_pred),
    }

    try:
        r = requests.post(SERVER_URL, data=payload, timeout=60)
        print("Resposta do servidor:\n", r.text)
        with open(RESPONSE_FILE, "w", encoding="utf-8") as f:
            f.write(r.text)
    except Exception as e:
        print("Falha ao enviar:", e)


if __name__ == "__main__":
    main()
