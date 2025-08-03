# Procesador de Imágenes de Libros

Este proyecto contiene un script de Python diseñado para automatizar el procesamiento por lotes de imágenes de libros abiertos. El script toma fotografías de dos páginas juntas, las corrige y las separa en archivos de imagen individuales para cada página.

## Características Principales

- **Procesamiento por Lotes:** Procesa miles de imágenes de forma automática.
- **Soporte de Formatos RAW:** Convierte y procesa archivos RAW de cámaras (`.CR3`, `.NEF`, `.ARW`) además de formatos estándar como `.JPG` y `.PNG`.
- **Corrección de Orientación:** Gira automáticamente las imágenes que están de cabeza (180°).
- **Enderezado Inteligente:** Detecta la inclinación del texto y endereza la página para que las líneas queden perfectamente horizontales.
- **Recorte en Dos Fases:**
  1.  **Recorte Global:** Aísla el libro del fondo (ej. la mesa).
  2.  **Recorte Fino:** Recorta cada página individualmente para eliminar los márgenes y dejar solo el texto.
- **División de Páginas:** Separa la imagen del libro abierto en dos archivos, uno para la página izquierda y otro para la derecha.
- **Procesamiento Paralelo:** Utiliza múltiples núcleos del procesador para acelerar el trabajo significativamente.
- **Registro de Errores:** Guarda un registro de cualquier imagen que no se pudo procesar en un archivo `errors.log` dentro de cada lote.

## Estructura del Proyecto

El proyecto espera la siguiente estructura de carpetas para funcionar correctamente:

```
contenedor/
└─ lote_X/                  # Carpeta para un lote de imágenes (puedes tener lote_1, lote_2, etc.)
   ├─ imagenes_juntas/       # Aquí debes colocar tus fotos originales.
   └─ imagenes_separadas/    # Aquí el script guardará las páginas procesadas.
```

## Requisitos

- Python 3.11 o superior.

## Instalación

Sigue estos pasos para preparar el entorno y dejar el script listo para funcionar.

1.  **Clonar el Repositorio**
    Si tienes Git, clona el repositorio. Si no, simplemente descarga los archivos del proyecto en una carpeta.

2.  **Crear un Entorno Virtual**
    Abre una terminal en la carpeta raíz del proyecto y ejecuta el siguiente comando para crear un entorno virtual. Esto aísla las dependencias del proyecto.

    ```bash
    python -m venv venv
    ```

3.  **Activar el Entorno Virtual**
    En la misma terminal, activa el entorno.

        - **En Windows:**
          ```powershell
          .\venv\Scripts\activate
          ```
        - **En macOS/Linux:**
          `bash

    source venv/bin/activate
    `      Verás`(venv)` al principio de la línea de tu terminal si se activó correctamente.

4.  **Instalar las Dependencias**
    Con el entorno activado, instala todas las librerías necesarias ejecutando:
    ```bash
    pip install -r requirements.txt
    ```

¡Y listo! El entorno está preparado.

## Uso

1.  **Coloca tus Imágenes:** Añade todas las fotos que quieras procesar dentro de la carpeta `contenedor/lote_1/imagenes_juntas/`. Puedes crear más carpetas de lotes si lo necesitas (ej. `lote_2`).

2.  **Ejecuta el Script:** Desde la terminal (con el entorno virtual activado), puedes ejecutar el script de dos maneras:

    - **Para procesar un único lote (ej. `lote_1`):**

      ```bash
      python process_book_pages.py --path contenedor/lote_1
      ```

    - **Para procesar todos los lotes (`lote_*`) que se encuentren dentro de `contenedor`:**

      ```bash
      python process_book_pages.py --path contenedor
      ```

    - **Opciones avanzadas disponibles:**

      ```bash
      # Usar canon_cr3 en lugar de rawpy (experimental)
      python process_book_pages.py --path contenedor/lote_1 --use-canon-cr3

      # Desactivar mejora de calidad
      python process_book_pages.py --path contenedor/lote_1 --no-enhance

      # Configuración personalizada de mejora
      python process_book_pages.py --path contenedor/lote_1 --sharpness 1.5 --contrast 1.3
      ```

3.  **Comparar Librerías CR3 (Opcional):** Si tienes archivos CR3 y quieres ver las diferencias entre `rawpy` y `canon_cr3`:

    ```bash
    python compare_libraries.py --cr3-file contenedor/lote_1/imagenes_juntas/archivo.CR3
    ```

    Esta herramienta te mostrará:

    - ⏱️ **Tiempo de procesamiento** de cada librería
    - 📦 **Tamaño de archivo** resultante
    - ✅ **Tasa de éxito** de conversión
    - 💡 **Recomendación** basada en rendimiento

4.  **Revisa los Resultados:** El script mostrará una barra de progreso. Cuando termine, encontrarás las imágenes de las páginas izquierda y derecha, corregidas y recortadas, dentro de la carpeta `imagenes_separadas` del lote correspondiente.

## Cómo Funciona el Procesamiento

Cada imagen pasa por un pipeline de 8 pasos para asegurar la máxima calidad en el resultado:

1.  **Carga:** Se carga la imagen (RAW o JPG) usando `rawpy` o `canon_cr3`.
2.  **Giro 180°:** Se corrige la orientación si está de cabeza.
3.  **Enderezado:** Se endereza la imagen para que el texto quede horizontal.
4.  **Recorte Global:** Se elimina el fondo (la mesa) para aislar solo el libro.
5.  **División:** Se divide el libro en página izquierda y derecha, con un ligero desplazamiento para evitar el lomo.
6.  **Recorte Fino:** Se recorta cada página para eliminar márgenes y dejar solo el texto.
7.  **Mejora de Calidad:** Se aplica nitidez y contraste optimizados para texto (opcional).
8.  **Guardado:** Se guardan las dos páginas finales como archivos JPG.

## Comparación de Librerías para Archivos CR3

El proyecto incluye soporte experimental para la librería `canon_cr3` además del estándar `rawpy`. Aquí están las diferencias:

### 📊 **rawpy vs canon_cr3**

| Aspecto            | rawpy ✅                      | canon_cr3 ⚠️                |
| ------------------ | ----------------------------- | --------------------------- |
| **Estabilidad**    | Muy alta, ampliamente probada | Experimental, puede fallar  |
| **Velocidad**      | Rápida                        | Variable                    |
| **Calidad**        | Excelente procesamiento RAW   | Extracción de previews JPEG |
| **Compatibilidad** | Todos los formatos RAW        | Solo CR3                    |
| **Mantenimiento**  | Activamente mantenida         | Desarrollo limitado         |
| **Recomendación**  | **Usar por defecto**          | Solo para pruebas           |

### 🔬 **¿Cuándo usar cada una?**

- **rawpy (recomendado):** Para producción, mayor calidad, estabilidad garantizada
- **canon_cr3:** Para experimentación, comparaciones técnicas, casos específicos

### 🧪 **Cómo comparar:**

```bash
# Procesar con rawpy (método estándar)
python process_book_pages.py --path contenedor/lote_1

# Procesar con canon_cr3 (experimental)
python process_book_pages.py --path contenedor/lote_1 --use-canon-cr3

# Comparación directa
python compare_libraries.py --cr3-file ruta/archivo.CR3
```

## Pruebas

El proyecto incluye pruebas unitarias para las funciones críticas. Para ejecutarlas, asegúrate de tener el entorno virtual activado y corre el siguiente comando:

```bash
pytest
```
