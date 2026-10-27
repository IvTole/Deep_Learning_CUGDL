import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Tema 12 · Detección y activaciones con webcam")


@app.cell
def _():
    import base64
    import io
    from pathlib import Path

    import marimo as mo
    import torch
    from PIL import Image
    from detection_webcam_utils import (
        DetectionWebcamWidget, LAYER_OPTIONS, activation_grid, annotate_detections,
        filter_detections, image_uri, infer_frame, load_detector, prepare_frame,
    )

    return (
        DetectionWebcamWidget, Image, LAYER_OPTIONS, Path, activation_grid,
        annotate_detections, base64, filter_detections, image_uri, infer_frame,
        io, load_detector, mo, prepare_frame, torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tema 12 · Detecta objetos y explora sus activaciones

    En la práctica de IoU dibujamos cajas a mano. Aquí un modelo preentrenado
    **SSDLite320 + MobileNetV3** propone cajas y clases para imágenes reales.
    Reutilizamos la idea de la webcam del tema 08, con una red de detección.

    **Objetivo:** relacionar imagen → características → cajas, clases y puntuaciones.
    No entrenaremos ni necesitamos anotar las capturas. El modelo fue entrenado con
    COCO y reconoce sus 80 categorías de objetos; no cualquier objeto que mostremos.
    Los nombres de categorías se conservan en inglés para coincidir con esos pesos.

    **¿Es pesado?** Este modelo tiene unos 3.4 millones de parámetros y sus pesos
    ocupan unos 13.4 MB. Empezaremos en CPU y con **una captura por segundo** como
    máximo. La frecuencia real depende del equipo y de la visualización; no es una
    promesa de video a 30 FPS. El primer cuadro puede tardar más por inicialización.

    1. Pulsa **Cargar detector**: descarga los pesos oficiales si aún no están en caché.
    2. Prueba la imagen de ejemplo o sube una imagen.
    3. Selecciona **Webcam**, activa la cámara y permite el acceso en el navegador.
    4. Captura un cuadro; después prueba actualización continua a 0.5–1 capturas/s.
    5. Compara detección completa con **Solo activaciones** y observa el tiempo medido.
    """)
    return


@app.cell
def _(mo, torch):
    device_options = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    selected_device = mo.ui.dropdown(device_options, value="cpu", label="Dispositivo")
    load_button = mo.ui.run_button(label="Cargar detector preentrenado")
    mo.vstack([
        mo.hstack([selected_device, load_button]),
        mo.md("CPU funciona también en Mac. Si hay CUDA puedes seleccionarla. "
              "No usamos MPS aquí porque la compatibilidad de las operaciones de "
              "postprocesamiento, como NMS, depende de la instalación."),
    ])
    return load_button, selected_device


@app.cell
def _(load_button, load_detector, mo, selected_device):
    mo.stop(not load_button.value, mo.md("Carga el modelo para habilitar la práctica."))
    try:
        detector, detector_weights = load_detector(selected_device.value)
    except (OSError, RuntimeError) as _error:
        mo.stop(True, mo.md(f"No se pudo cargar el modelo: `{_error}`. "
                            "Revisa la conexión para la primera descarga y vuelve a pulsar Cargar."))
    mo.md(f"**SSDLite listo en {selected_device.value}** · "
          f"{sum(_p.numel() for _p in detector.parameters()):,} parámetros · modo evaluación.")
    return detector, detector_weights


@app.cell
def _(mo):
    source = mo.ui.dropdown(["Imagen de ejemplo", "Archivo", "Webcam"],
                            value="Imagen de ejemplo", label="Fuente")
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"],
                        multiple=False, label="Subir imagen")
    mo.hstack([source, upload], wrap=True)
    return source, upload


@app.cell
def _(DetectionWebcamWidget, mo):
    webcam_widget = DetectionWebcamWidget()
    camera = mo.ui.anywidget(webcam_widget)
    return camera, webcam_widget


@app.cell
def _(source, webcam_widget):
    # Al abandonar Webcam se detienen también las pistas de la cámara en el navegador.
    webcam_widget.enabled = source.value == "Webcam"
    return


@app.cell
def _(camera, detector, mo, source):
    mo.stop(source.value != "Webcam")
    mo.vstack([
        camera,
        mo.md("Empieza con captura manual. Solo se envía un cuadro nuevo cuando Python "
              "termina el anterior; si el modelo tarda, no se acumulan capturas. "
              "Pulsa Detener cámara al terminar; cambiar de fuente también la detiene."),
    ])
    return


@app.cell
def _(LAYER_OPTIONS, mo):
    inference_mode = mo.ui.dropdown(
        {"Detección + activaciones": "both", "Solo detección": "detections",
         "Solo activaciones (salida temprana)": "features"},
        value="Detección + activaciones", label="Trabajo a realizar",
    )
    layer_choice = mo.ui.dropdown(
        options=list(LAYER_OPTIONS), value=next(iter(LAYER_OPTIONS)), label="Capa observada",
    )
    map_count = mo.ui.slider(1, 16, value=8, show_value=True, label="Mapas a copiar y mostrar")
    mo.hstack([inference_mode, layer_choice, map_count], wrap=True)
    return inference_mode, layer_choice, map_count


@app.cell
def _(LAYER_OPTIONS, layer_choice, mo):
    _channels = LAYER_OPTIONS[layer_choice.value][1]
    channel_start = mo.ui.slider(0, _channels - 1, value=0, show_value=True,
                                 label="Canal inicial (se reinicia al cambiar capa)")
    color_limit = mo.ui.slider(0.5, 10, step=0.5, value=3, show_value=True,
                               label="Límite absoluto de color")
    mo.hstack([channel_start, color_limit])
    return channel_start, color_limit


@app.cell
def _(mo):
    score_threshold = mo.ui.slider(0.05, 0.95, step=0.05, value=0.5,
                                   show_value=True, label="Umbral de confianza")
    class_filter = mo.ui.dropdown(
        {"Todas las clases": None, "Solo personas": 1}, value="Todas las clases", label="Mostrar",
    )
    max_boxes = mo.ui.slider(1, 30, value=15, show_value=True, label="Máximo de cajas visibles")
    mo.hstack([score_threshold, class_filter, max_boxes], wrap=True)
    return class_filter, max_boxes, score_threshold


@app.cell
def _(Image, Path, base64, camera, io, mo, prepare_frame, source, upload, webcam_widget):
    frame_sequence = None
    if source.value == "Webcam":
        _frame = camera.frame
        mo.stop(not _frame, mo.md("Activa la cámara y captura un cuadro."))
        frame_sequence = int(_frame["seq"])
        try:
            with Image.open(io.BytesIO(base64.b64decode(_frame["data"].split(",", 1)[1]))) as _image:
                raw_image = prepare_frame(_image)
        except (OSError, ValueError, IndexError) as _error:
            webcam_widget.processed_seq = frame_sequence
            mo.stop(True, mo.md(f"No se pudo decodificar la captura: `{_error}`"))
        frame_label = f"Webcam · cuadro {frame_sequence}"
    elif source.value == "Archivo":
        mo.stop(not upload.value, mo.md("Sube una imagen para continuar."))
        try:
            with Image.open(io.BytesIO(upload.value[0].contents)) as _image:
                raw_image = prepare_frame(_image)
        except (OSError, ValueError) as _error:
            mo.stop(True, mo.md(f"No se pudo abrir la imagen: `{_error}`"))
        frame_label = "Archivo subido"
    else:
        _path = Path(mo.notebook_location()).parent / "happy_dog.jpg"
        mo.stop(not _path.is_file(), mo.md("No está `happy_dog.jpg`; selecciona Archivo o Webcam."))
        with Image.open(_path) as _image:
            raw_image = prepare_frame(_image)
        frame_label = "Ejemplo local · happy_dog.jpg"
    return frame_label, frame_sequence, raw_image


@app.cell
def _(
    LAYER_OPTIONS, channel_start, detector, detector_weights, frame_sequence,
    infer_frame, inference_mode, layer_choice, map_count, mo, raw_image, webcam_widget,
):
    try:
        frame_result = infer_frame(
            detector, detector_weights, raw_image, mode=inference_mode.value,
            layer_name=LAYER_OPTIONS[layer_choice.value][0],
            first_channel=channel_start.value, channel_count=map_count.value,
        )
    except (RuntimeError, ValueError) as _error:
        mo.stop(True, mo.md(f"Falló la inferencia: `{_error}`. Prueba CPU o recarga el modelo."))
    finally:
        if frame_sequence is not None:
            webcam_widget.processed_seq = frame_sequence
    return (frame_result,)


@app.cell
def _(
    annotate_detections, class_filter, detector_weights, filter_detections, frame_label,
    frame_result, image_uri, max_boxes, mo, raw_image, score_threshold,
):
    _prediction = frame_result["detections"]
    if _prediction is None:
        _picture = raw_image
        _rows = []
        _caption = "Salida temprana: esta pasada no ejecutó las cabezas ni produjo cajas."
    else:
        _filtered = filter_detections(_prediction, score_threshold.value, class_filter.value, max_boxes.value)
        _picture, _rows = annotate_detections(raw_image, _filtered, detector_weights.meta["categories"])
        _caption = (f"{len(_rows)} detecciones visibles después del umbral y filtro de clase. "
                    "Las cajas se expresan sobre la imagen mostrada, de lado máximo 640 px.")
    mo.vstack([
        mo.md(f"### {frame_label}\n\n{_caption}\n\n"
              f"**Tiempo de cómputo medido:** {frame_result['elapsed_ms']:.1f} ms. "
              "Incluye el forward, postprocesamiento si hay detección y copia de mapas; "
              "excluye captura, transferencia desde el navegador y dibujo. No equivale a FPS del video."),
        mo.Html(f'<img src="{image_uri(_picture)}" style="width:100%;max-width:800px" />'),
        mo.ui.table(_rows) if _rows else mo.md("No hay cajas que mostrar con esta configuración."),
    ])
    return


@app.cell
def _(LAYER_OPTIONS, activation_grid, color_limit, frame_result, layer_choice, mo):
    _maps = frame_result["maps"]
    mo.stop(_maps is None, mo.md("Activa un modo con activaciones para explorar las capas."))
    mo.vstack([
        mo.md(f"### {LAYER_OPTIONS[layer_choice.value][0]}\n\n"
              f"Salida completa: `{frame_result['shape']}` (batch, canales, alto, ancho). "
              f"Se copiaron y muestran **{len(_maps)} canales**.\n\n"
              f"Escala fija entre cuadros: **−{color_limit.value:g} a +{color_limit.value:g}**. "
              "Azul: negativo; blanco: cercano a cero; rojo: positivo. "
              "Los valores fuera de escala se saturan. Cambiarla no modifica la red."),
        mo.Html(activation_grid(_maps, frame_result["first_channel"], color_limit.value)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Qué estamos calculando?

    ```text
    RGB [0,1] → normalización y resize internos a 320×320 → MobileNetV3
                 → mapas a varias escalas → cabezas de clase y caja → NMS → detecciones
    ```

    **Normalización:** `weights.transforms()` convierte la imagen a RGB tensor en
    [0,1]. El detector realiza internamente su propia normalización y resize;
    no añadimos la normalización de ImageNet del notebook de VGG16. Normalizar
    otra vez alteraría la distribución esperada. Conservamos la imagen RGB para dibujar.

    **Mapas:** cada cuadro es la respuesta de un canal de un bloque, no un kernel,
    una máscara de segmentación ni una explicación causal de una detección. En
    MobileNetV3 hay convoluciones, conexiones residuales y activaciones como Hardswish,
    por lo que no todos los mapas son estrictamente positivos. La resolución disminuye
    al avanzar por el backbone; no significa que cada canal corresponda a una clase.
    Los mapas observan la entrada interna de 320×320, no la resolución original.

    **Detección + activaciones:** un hook temporal copia solo los canales solicitados
    durante la misma pasada de detección; se elimina al terminar, incluso si hay error.
    No ejecutamos un segundo forward para obtener los mapas.

    **Solo activaciones:** se ejecuta el mismo preprocesamiento y el prefijo del
    backbone hasta la capa seleccionada. Se omiten los bloques posteriores y las
    cabezas de detección. Por eso puede ser más rápido, pero no devuelve cajas.
    Mostrar menos mapas por sí solo reduce copia y dibujo, no el costo del backbone.

    **Confianza y NMS:** el modelo ya devuelve detecciones después de NMS, que usa IoU
    entre cajas candidatas para reducir duplicados. Nuestro control filtra por score
    esas salidas; no cambia el umbral de NMS ni recupera cajas ya eliminadas.
    Sin anotaciones reales no podemos calcular la IoU de localización ni accuracy
    sobre la webcam. Los scores altos tampoco garantizan acierto.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experimentos para clase

    1. Muestra una persona, botella o taza. Cambia distancia e iluminación: ¿se
       conservan las detecciones? ¿Qué ocurre con objetos pequeños?
    2. Baja el umbral de confianza de 0.7 a 0.2 sobre **la misma captura**. Registra
       cambios en cantidad de cajas y falsos positivos aparentes. No necesitas recapturar.
    3. Compara una capa inicial y una profunda: registra sus formas y observa cuánto
       detalle espacial se conserva. No asignes nombres semánticos a los canales sin evidencia.
    4. Fija la primera capa y mide detección completa frente a salida temprana.
       Descarta la primera medida y repite cinco veces con capturas comparables.
       Reporta la mediana y distingue tiempo de forward de fluidez en el navegador.
    5. Prueba un objeto fuera de las categorías COCO. ¿Por qué una etiqueta incorrecta
       con score alto no demuestra que el modelo haya aprendido esa categoría nueva?

    **Privacidad y funcionamiento:** la cámara se solicita únicamente al activarla.
    Las capturas se envían al proceso Python que ejecuta este notebook; no se guardan
    imágenes en disco ni se usa un servicio externo para inferencia. Si el notebook
    está en un servidor remoto, las capturas viajan a ese servidor. El acceso a cámara
    requiere localhost o HTTPS. Los pesos se descargan desde PyTorch la primera vez.

    **Si va lento:** usa CPU con captura manual o 0.5–1 capturas/s, 4–8 mapas y la capa
    inicial. Desactiva mapas si solo necesitas cajas. Para estudiar características
    con menor cómputo, elige salida temprana. Una GPU CUDA disponible puede acelerar
    el forward; la captura y el dibujo también influyen en la experiencia.

    [Modelo, pesos y preprocesamiento oficiales de SSDLite](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)
    """)
    return


if __name__ == "__main__":
    app.run()
