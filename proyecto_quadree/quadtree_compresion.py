"""
Compresión de imágenes mediante Quadtree y Divide y Vencerás.

Instalación:
    pip install -r requirements.txt

Ejecutar todos los experimentos:
    python quadtree_compresion.py

Comprimir una imagen propia:
    python quadtree_compresion.py --image "mi_imagen.png" --tolerance 10

La construcción del Quadtree y la reconstrucción están implementadas aquí.
Pillow, NumPy, Pandas y Matplotlib se usan como bibliotecas auxiliares.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


OUTPUT_DIR = Path("resultados_quadtree")


@dataclass
class QuadNode:
    """Nodo del Quadtree. Las coordenadas finales son exclusivas."""
    x0: int
    y0: int
    x1: int
    y1: int
    color: Optional[tuple[int, ...]] = None
    children: Optional[list["QuadNode"]] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


def load_image(path: str) -> tuple[np.ndarray, str]:
    """Carga una imagen y la convierte a RGB o escala de grises."""
    image = Image.open(path)
    if image.mode in ("1", "L", "I", "F"):
        array = np.asarray(image.convert("L"), dtype=np.uint8)
        mode = "L"
    else:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mode = "RGB"
    return array, mode


def region_is_uniform(region: np.ndarray, tolerance: float) -> tuple[bool, tuple[int, ...]]:
    """
    Una región es uniforme si el mayor desvío absoluto de cualquier canal
    respecto al color promedio de la región no supera la tolerancia.
    """
    pixels = region.astype(np.float64)
    axes = (0, 1)
    mean_color = pixels.mean(axis=axes)
    max_deviation = float(np.max(np.abs(pixels - mean_color)))
    representative = tuple(int(v) for v in np.clip(np.rint(mean_color), 0, 255))
    return max_deviation <= tolerance, representative


def build_quadtree(
    image: np.ndarray,
    tolerance: float,
    x0: int = 0,
    y0: int = 0,
    x1: Optional[int] = None,
    y1: Optional[int] = None,
) -> QuadNode:
    """
    Construye el árbol recursivamente.
    Divide cada región no uniforme en hasta cuatro subregiones no vacías.
    Las coordenadas x1/y1 son exclusivas, por lo que admite dimensiones impares.
    """
    if x1 is None:
        x1 = image.shape[1]
    if y1 is None:
        y1 = image.shape[0]

    region = image[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError("No se permite construir nodos para regiones vacías.")

    uniform, color = region_is_uniform(region, tolerance)
    node = QuadNode(x0, y0, x1, y1)

    # Caso base: región uniforme o un único píxel.
    if uniform or (x1 - x0 == 1 and y1 - y0 == 1):
        node.color = color
        return node

    # División espacial: el uso de límites exclusivos cubre dimensiones impares.
    xm = x0 + (x1 - x0) // 2
    ym = y0 + (y1 - y0) // 2
    boxes = [
        (x0, y0, xm, ym),  # superior izquierda
        (xm, y0, x1, ym),  # superior derecha
        (x0, ym, xm, y1),  # inferior izquierda
        (xm, ym, x1, y1),  # inferior derecha
    ]

    children: list[QuadNode] = []
    for ax0, ay0, ax1, ay1 in boxes:
        if ax0 < ax1 and ay0 < ay1:
            children.append(build_quadtree(image, tolerance, ax0, ay0, ax1, ay1))

    # Si no fue posible dividir (protección adicional), convertir en hoja.
    if not children:
        node.color = color
        return node

    node.children = children
    return node


def reconstruct(node: QuadNode, output: np.ndarray) -> None:
    """Reconstruye la imagen pintando cada región hoja con su color."""
    if node.is_leaf:
        output[node.y0:node.y1, node.x0:node.x1] = node.color
        return
    for child in node.children or []:
        reconstruct(child, output)


def reconstruct_image(root: QuadNode, original: np.ndarray) -> np.ndarray:
    """Crea y devuelve una matriz con las dimensiones de la imagen original."""
    result = np.zeros_like(original)
    reconstruct(root, result)
    return result


def tree_statistics(root: QuadNode) -> dict[str, int]:
    """Calcula número de nodos, hojas y profundidad máxima."""
    total = 0
    leaves = 0
    max_depth = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        total += 1
        max_depth = max(max_depth, depth)
        if node.is_leaf:
            leaves += 1
        else:
            stack.extend((child, depth + 1) for child in node.children or [])
    return {
        "total_nodes": total,
        "leaf_nodes": leaves,
        "max_depth": max_depth,
    }


def estimate_storage_bits(root: QuadNode, image: np.ndarray, mode: str) -> int:
    """
    Formato estimado:
    - Cabecera: 32 bits de ancho + 32 de alto + 1 bit para modo RGB/gris.
    - Cada nodo: 1 bit (hoja/interno).
    - Cada hoja: 8 bits por canal para su color.
    Las dimensiones permiten inferir la región de cada nodo; no se almacenan
    coordenadas por nodo. No incluye cabeceras propias de un archivo contenedor.
    """
    stats = tree_statistics(root)
    channels = 3 if mode == "RGB" else 1
    header_bits = 32 + 32 + 1
    node_bits = stats["total_nodes"]
    color_bits = stats["leaf_nodes"] * channels * 8
    return header_bits + node_bits + color_bits


def calculate_metrics(
    original: np.ndarray, reconstructed: np.ndarray, root: QuadNode, mode: str
) -> dict[str, float | int]:
    """Calcula MSE, PSNR y razón de compresión según el formato estimado."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10 * math.log10((255.0 ** 2) / mse))

    original_bits = int(original.size * 8)
    compressed_bits = estimate_storage_bits(root, original, mode)
    ratio = original_bits / compressed_bits if compressed_bits else float("inf")
    reduction_percent = (1.0 - compressed_bits / original_bits) * 100.0

    stats = tree_statistics(root)
    return {
        **stats,
        "mse": mse,
        "psnr_db": psnr,
        "original_bits": original_bits,
        "estimated_compressed_bits": compressed_bits,
        "compression_ratio": ratio,
        "storage_reduction_percent": reduction_percent,
    }


def make_test_images(size: int) -> dict[str, np.ndarray]:
    """Genera tres imágenes reproducibles: uniforme, regiones y detalles."""
    # Imagen uniforme RGB.
    uniform = np.zeros((size, size, 3), dtype=np.uint8)
    uniform[:] = (70, 130, 190)

    # Imagen con regiones de colores definidos.
    regions = np.zeros((size, size, 3), dtype=np.uint8)
    regions[: size // 2, : size // 2] = (220, 50, 50)
    regions[: size // 2, size // 2 :] = (50, 200, 70)
    regions[size // 2 :, : size // 2] = (50, 80, 220)
    regions[size // 2 :, size // 2 :] = (240, 210, 60)

    # Imagen con detalles y variación de color, generada con semilla fija.
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=np.float64)
    base[:, :, 0] = 127 + 70 * np.sin(xx / 7.0)
    base[:, :, 1] = 127 + 70 * np.cos(yy / 9.0)
    base[:, :, 2] = 127 + 60 * np.sin((xx + yy) / 11.0)
    rng = np.random.default_rng(2026 + size)
    noise = rng.normal(0, 12, size=(size, size, 3))
    detailed = np.clip(base + noise, 0, 255).astype(np.uint8)

    return {"uniforme": uniform, "regiones": regions, "detalles": detailed}


def save_image(array: np.ndarray, path: Path) -> None:
    """Guarda una matriz RGB o de escala de grises como imagen PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(path)
    else:
        Image.fromarray(array, mode="RGB").save(path)


def run_one(
    name: str, image: np.ndarray, tolerance: float, output_dir: Path, mode: str = "RGB"
) -> dict[str, object]:
    """Ejecuta una compresión, mide el tiempo y guarda original/reconstruida."""
    start = time.perf_counter()
    tree = build_quadtree(image, tolerance)
    elapsed = time.perf_counter() - start
    rebuilt = reconstruct_image(tree, image)
    metrics = calculate_metrics(image, rebuilt, tree, mode)

    safe_name = name.replace(" ", "_")
    prefix = f"{safe_name}_tol_{tolerance:g}"
    save_image(image, output_dir / f"{prefix}_original.png")
    save_image(rebuilt, output_dir / f"{prefix}_reconstruida.png")

    return {
        "imagen": name,
        "alto": int(image.shape[0]),
        "ancho": int(image.shape[1]),
        "canales": 1 if image.ndim == 2 else 3,
        "tolerancia": tolerance,
        "tiempo_construccion_s": elapsed,
        **metrics,
        "archivo_original": f"{prefix}_original.png",
        "archivo_reconstruido": f"{prefix}_reconstruida.png",
    }


def run_experiments(output_dir: Path) -> pd.DataFrame:
    """Corre 3 tamaños x 3 contenidos x 3 tolerancias y exporta métricas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    tolerances = [0, 10, 25]
    sizes = [64, 128, 256]

    for size in sizes:
        images = make_test_images(size)
        for image_type, array in images.items():
            name = f"{image_type}_{size}x{size}"
            for tolerance in tolerances:
                print(f"Procesando {name}, tolerancia={tolerance}...")
                rows.append(run_one(name, array, tolerance, output_dir, "RGB"))

    df = pd.DataFrame(rows)
    csv_path = output_dir / "resultados_experimentales.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    create_graphs(df, output_dir)
    return df


def create_graphs(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera gráficas de tiempo, nodos, compresión y error."""
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_specs = [
        ("tiempo_construccion_s", "Tiempo de construcción (s)", "tiempo.png"),
        ("total_nodes", "Número total de nodos", "nodos.png"),
        ("compression_ratio", "Razón de compresión (original/comprimido)", "compresion.png"),
        ("mse", "Error cuadrático medio (MSE)", "mse.png"),
    ]
    for column, ylabel, filename in graph_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        for image_name, group in df.groupby("imagen"):
            ordered = group.sort_values("tolerancia")
            ax.plot(
                ordered["tolerancia"],
                ordered[column],
                marker="o",
                label=image_name,
            )
        ax.set_xlabel("Tolerancia")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " según la tolerancia")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


def compress_user_image(image_path: str, tolerance: float, output_dir: Path) -> None:
    """Comprime una imagen indicada por el usuario y muestra sus métricas."""
    array, mode = load_image(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(image_path).stem
    result = run_one(name, array, tolerance, output_dir, mode)
    print("\nRESULTADOS")
    print("-" * 45)
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"\nArchivos guardados en: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compresión de imágenes con Quadtree y Divide y Vencerás."
    )
    parser.add_argument("--image", type=str, help="Ruta a una imagen local.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10,
        help="Tolerancia de color (0 exige uniformidad exacta).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="Carpeta de salida (por defecto: resultados_quadtree).",
    )
    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.tolerance < 0:
        parser.error("--tolerance no puede ser negativa.")

    if args.image:
        compress_user_image(args.image, args.tolerance, output_dir)
    else:
        df = run_experiments(output_dir)
        print("\nEXPERIMENTOS TERMINADOS")
        print(f"CSV: {(output_dir / 'resultados_experimentales.csv').resolve()}")
        print(f"Gráficas e imágenes: {output_dir.resolve()}")
        print("\nVista de algunas métricas:")
        columns = [
            "imagen", "tolerancia", "tiempo_construccion_s",
            "total_nodes", "leaf_nodes", "max_depth", "mse",
            "compression_ratio", "storage_reduction_percent",
        ]
        print(df[columns].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
