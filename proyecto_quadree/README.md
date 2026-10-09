# Compresión de imágenes mediante Quadtree y Divide y Vencerás

## Descripción

Este proyecto implementa la compresión de imágenes mediante un **Quadtree**, usando una estrategia recursiva de divide y vencerás. Cada región de la imagen se evalúa según la variación de sus colores: si cumple con la tolerancia indicada, se representa mediante un color promedio; de lo contrario, se divide en hasta cuatro subregiones.

El programa reconstruye la imagen a partir del árbol y calcula métricas para evaluar el resultado, como el error cuadrático medio (MSE), la relación señal-ruido pico (PSNR), la razón de compresión y la reducción estimada de almacenamiento.

## Requisitos

- Python 3.10 o posterior recomendado.
- `Pillow`
- `NumPy`
- `Pandas`
- `Matplotlib`

## Instalación

1. Descarga o clona el proyecto y abre una terminal en la carpeta que contiene `quadtree_compresion.py`.
2. (Opcional, recomendado) Crea un entorno virtual:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS o Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias.

   Si el proyecto incluye un archivo `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   Si todavía no tienes ese archivo, instala las bibliotecas directamente:
   ```bash
   python -m pip install Pillow numpy pandas matplotlib
   ```

## Ejecución

### Ejecutar todos los experimentos

Desde la carpeta del proyecto, ejecuta:

```bash
python quadtree_compresion.py
```

El programa genera imágenes de prueba con distintos contenidos, tamaños y tolerancias. Al finalizar, muestra una vista de algunas métricas en la terminal.

Los experimentos usan tamaños de 64 × 64, 128 × 128 y 256 × 256 píxeles, y tolerancias de 0, 10 y 25.

### Comprimir una imagen propia

Indica la ruta de la imagen y la tolerancia deseada:

```bash
python quadtree_compresion.py --image "mi_imagen.png" --tolerance 10
```

Reemplaza `mi_imagen.png` por la ruta y el nombre reales de tu archivo. Por ejemplo, si la imagen está en una carpeta llamada `imagenes` dentro del proyecto:

```bash
python quadtree_compresion.py --image "imagenes/foto.png" --tolerance 10
```

La tolerancia es un número no negativo. Una tolerancia de `0` exige uniformidad exacta en cada región; al aumentar la tolerancia, más regiones pueden representarse con un solo color, aunque puede aumentar la diferencia respecto de la imagen original.

### Elegir otra carpeta de salida

Puedes especificar dónde guardar los resultados mediante `--output`:

```bash
python quadtree_compresion.py --image "mi_imagen.png" --tolerance 10 --output "mis_resultados"
```

Si no indicas una carpeta, se usa `resultados_quadtree`.

## Archivos generados

En la carpeta de salida se guardan:

- `*_original.png`: imagen original procesada.
- `*_reconstruida.png`: imagen reconstruida a partir del Quadtree.
- `resultados_experimentales.csv`: tabla de métricas de todos los experimentos.
- `tiempo.png`: gráfica del tiempo de construcción.
- `nodos.png`: gráfica del número total de nodos.
- `compresion.png`: gráfica de la razón de compresión.
- `mse.png`: gráfica del error cuadrático medio.

Al comprimir una imagen propia, también se muestran en la terminal las métricas de esa ejecución.

## Métricas

- **MSE:** mide el error cuadrático medio entre la imagen original y la reconstruida. Un valor menor indica una diferencia menor.
- **PSNR:** expresa la relación señal-ruido pico en decibelios. Cuando el MSE es cero, el programa reporta infinito.
- **Razón de compresión:** compara el tamaño original estimado con el tamaño comprimido estimado. Un valor mayor que 1 indica que el tamaño estimado comprimido es menor.
- **Reducción de almacenamiento:** porcentaje estimado de reducción según el formato descrito en el código.
- **Nodos, hojas y profundidad máxima:** describen la estructura del Quadtree.
- **Tiempo de construcción:** tiempo empleado en construir el árbol.

**Nota:** el tamaño comprimido es una estimación del formato definido en el programa; no corresponde necesariamente al tamaño real de un archivo comprimido en disco. Las imágenes se guardan como PNG para facilitar su inspección.

## Solución de problemas

- **No se encuentra Python:** comprueba la instalación y prueba `python --version` o `python3 --version`.
- **Falta una biblioteca:** instala las dependencias con `python -m pip install Pillow numpy pandas matplotlib`, o usa el archivo `requirements.txt` si está disponible.
- **No se encuentra la imagen:** comprueba la ruta y el nombre del archivo. Puedes usar una ruta absoluta entre comillas.
- **La tolerancia es negativa:** el programa no la acepta; usa un valor igual o mayor que cero.

## Estructura mínima del proyecto

```text
proyecto/
├── quadtree_compresion.py
├── README.md
└── requirements.txt   # opcional si aún no está creado
```

## Autoría y propósito académico

Proyecto académico para estudiar la compresión y reconstrucción de imágenes mediante estructuras Quadtree y la estrategia de divide y vencerás.
