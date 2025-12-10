"""
Analiza el dataset OCT y muestra estadísticas de calidad.

Uso:
    python scripts/analyze_oct_dataset.py --root dataset/oct/train
    python scripts/analyze_oct_dataset.py --root dataset/oct/train --visualize
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


def analyze_oct_image(img_path: Path) -> dict:
    """Analiza una imagen OCT y retorna métricas."""
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    h, w = arr.shape

    # Métricas básicas
    mean_brightness = arr.mean()
    std_contrast = arr.std()

    # Análisis vertical
    row_brightness = arr.mean(axis=1)
    content_rows = row_brightness > 15

    if content_rows.any():
        first_content = np.argmax(content_rows)
        last_content = h - np.argmax(content_rows[::-1]) - 1
        top_black = first_content / h
        bottom_black = (h - last_content - 1) / h
        content_height = (last_content - first_content) / h
    else:
        top_black = bottom_black = 1.0
        content_height = 0.0

    # Estructura de capas (gradiente vertical en región central)
    y_start, y_end = int(h * 0.15), int(h * 0.85)
    x_start, x_end = int(w * 0.1), int(w * 0.9)
    center = arr[y_start:y_end, x_start:x_end].astype(np.float32)
    v_gradient = np.abs(np.diff(center, axis=0)).mean()
    layer_contrast = center.mean(axis=1).std()

    # Centrado
    bright_mask = arr > 30
    if bright_mask.sum() > 100:
        y_coords, _ = np.where(bright_mask)
        center_y_offset = abs(y_coords.mean() / h - 0.5)
    else:
        center_y_offset = 1.0

    return {
        "path": img_path,
        "brightness": mean_brightness,
        "contrast": std_contrast,
        "top_black": top_black,
        "bottom_black": bottom_black,
        "content_height": content_height,
        "v_gradient": v_gradient,
        "layer_contrast": layer_contrast,
        "center_offset": center_y_offset,
    }


def classify_quality(metrics: dict) -> str:
    """Clasifica la calidad de la imagen OCT."""
    issues = []

    if metrics["top_black"] > 0.35:
        issues.append("exceso_negro_arriba")
    if metrics["bottom_black"] > 0.40:
        issues.append("exceso_negro_abajo")
    if metrics["content_height"] < 0.35:
        issues.append("poca_altura_contenido")
    if metrics["v_gradient"] < 8.0:
        issues.append("sin_capas_visibles")
    if metrics["layer_contrast"] < 25.0:
        issues.append("bajo_contraste_capas")
    if metrics["center_offset"] > 0.25:
        issues.append("descentrado")
    if metrics["contrast"] < 15.0:
        issues.append("bajo_contraste")

    if not issues:
        return "BUENA"
    elif len(issues) <= 2:
        return f"ACEPTABLE ({', '.join(issues)})"
    else:
        return f"MALA ({', '.join(issues)})"


def main():
    parser = argparse.ArgumentParser(description="Analiza dataset OCT")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--visualize", action="store_true", help="Muestra ejemplos")
    parser.add_argument("--limit", type=int, default=None, help="Limitar análisis")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"No encontrado: {args.root}")
        return

    # Buscar imágenes
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".bmp")
    images = sorted(p for p in args.root.rglob("*") if p.suffix.lower() in extensions)

    if args.limit:
        images = images[:args.limit]

    print(f"Analizando {len(images)} imágenes OCT...")
    print("=" * 70)

    all_metrics = []
    quality_counts = {"BUENA": 0, "ACEPTABLE": 0, "MALA": 0}

    for img_path in images:
        try:
            m = analyze_oct_image(img_path)
            quality = classify_quality(m)
            m["quality"] = quality
            all_metrics.append(m)

            if "BUENA" in quality:
                quality_counts["BUENA"] += 1
            elif "ACEPTABLE" in quality:
                quality_counts["ACEPTABLE"] += 1
            else:
                quality_counts["MALA"] += 1

        except Exception as e:
            print(f"Error: {img_path.name}: {e}")

    # Estadísticas
    print(f"\n📊 RESUMEN DE CALIDAD")
    print(f"   Total: {len(all_metrics)}")
    print(f"   ✅ Buenas: {quality_counts['BUENA']} ({100*quality_counts['BUENA']/len(all_metrics):.1f}%)")
    print(f"   ⚠️  Aceptables: {quality_counts['ACEPTABLE']} ({100*quality_counts['ACEPTABLE']/len(all_metrics):.1f}%)")
    print(f"   ❌ Malas: {quality_counts['MALA']} ({100*quality_counts['MALA']/len(all_metrics):.1f}%)")

    # Promedios
    print(f"\n📈 MÉTRICAS PROMEDIO")
    for key in ["brightness", "contrast", "top_black", "bottom_black", "content_height", "v_gradient", "layer_contrast"]:
        values = [m[key] for m in all_metrics]
        print(f"   {key}: {np.mean(values):.2f} (std: {np.std(values):.2f})")

    # Mostrar ejemplos de cada categoría
    print(f"\n📷 EJEMPLOS POR CATEGORÍA")
    for cat in ["BUENA", "MALA"]:
        examples = [m for m in all_metrics if cat in m["quality"]][:3]
        print(f"\n   {cat}:")
        for m in examples:
            print(f"      {m['path'].name}")
            print(f"         top_black={m['top_black']:.2f}, bottom={m['bottom_black']:.2f}, layers={m['v_gradient']:.1f}")

    # Visualización
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle("Ejemplos de OCT: Arriba=Buenas, Abajo=Malas")

            good = [m for m in all_metrics if "BUENA" in m["quality"]][:4]
            bad = [m for m in all_metrics if "MALA" in m["quality"]][:4]

            for i, m in enumerate(good[:4]):
                img = Image.open(m["path"])
                axes[0, i].imshow(img, cmap="gray")
                axes[0, i].set_title(f"top={m['top_black']:.2f}")
                axes[0, i].axis("off")

            for i, m in enumerate(bad[:4]):
                img = Image.open(m["path"])
                axes[1, i].imshow(img, cmap="gray")
                axes[1, i].set_title(f"top={m['top_black']:.2f}")
                axes[1, i].axis("off")

            plt.tight_layout()
            out_path = args.root.parent / "oct_quality_analysis.png"
            plt.savefig(out_path, dpi=150)
            print(f"\n💾 Visualización guardada: {out_path}")
            plt.show()

        except ImportError:
            print("Instala matplotlib para visualizar: pip install matplotlib")


if __name__ == "__main__":
    main()
