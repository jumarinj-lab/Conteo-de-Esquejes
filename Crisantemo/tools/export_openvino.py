from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "modelos"
DEFAULT_IMGSZ = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta un modelo .pt a OpenVINO y opcionalmente actualiza la carpeta estandar best_openvino_model."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="best26V2.pt",
        help="Nombre del archivo .pt dentro de la carpeta modelos o ruta absoluta al modelo.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Tamano de imagen para la exportacion.",
    )
    parser.add_argument(
        "--standard-name",
        default="best_openvino_model",
        help="Nombre de la carpeta estandar que usan los scripts de inferencia.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="No reemplaza la carpeta estandar al final de la exportacion.",
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> Path:
    model_path = Path(model_arg)
    if not model_path.is_absolute():
        model_path = MODELS_DIR / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"El modelo debe ser .pt: {model_path}")
    return model_path


def export_model(model_path: Path, imgsz: int) -> Path:
    exported_path = Path(YOLO(str(model_path)).export(format="openvino", imgsz=imgsz))
    return exported_path.resolve()


def standardize_export(exported_dir: Path, standard_name: str) -> Path:
    standard_dir = MODELS_DIR / standard_name
    backup_dir = MODELS_DIR / f"{standard_name}_backup"

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if standard_dir.exists():
        shutil.move(str(standard_dir), str(backup_dir))

    shutil.copytree(exported_dir, standard_dir)
    return standard_dir


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)

    print(f"[INFO] Exportando modelo: {model_path.name}")
    exported_dir = export_model(model_path, args.imgsz)
    print(f"[INFO] Export creado en: {exported_dir}")

    if args.no_standardize:
        print("[INFO] Exportacion terminada sin actualizar carpeta estandar.")
        return

    standard_dir = standardize_export(exported_dir, args.standard_name)
    print(f"[INFO] Carpeta estandar actualizada: {standard_dir}")


if __name__ == "__main__":
    main()
