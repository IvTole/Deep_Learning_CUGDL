# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.25.1", "torch", "numpy", "pillow", "matplotlib", "anywidget", "traitlets"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="CNN · Webcam y mapas de activación")


@app.cell
def _():
    import marimo as mo
    import anywidget
    import traitlets
    import base64
    import io
    from pathlib import Path
    from PIL import Image
    from cnn_webcam_utils import (
        load_cnn, prepare_image, extract_activations, mnist_sample,
        image_uri, activation_grid,
    )

    return (
        Image,
        Path,
        activation_grid,
        anywidget,
        base64,
        extract_activations,
        image_uri,
        io,
        load_cnn,
        mnist_sample,
        mo,
        prepare_image,
        traitlets,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Mira cómo responde una CNN a una imagen

    Cargamos los **pesos ya entrenados** del tema 08. Cada cuadro del grid es
    un **mapa de activación**: la respuesta de un canal a la imagen actual.
    Los pesos permanecen fijos; aquí no entrenamos.

    Empieza con **MNIST local** para explorar las capas. Después selecciona
    **Webcam**, activa la cámara y muestra un dígito escrito en una hoja.
    Para una hoja blanca con tinta oscura, activa **Invertir colores** y
    **Centrar dígito**. Observa la entrada de 28×28 para ajustar el encuadre.
    """)
    return


@app.cell
def _(Path, mo):
    notebook_root = Path(__file__).resolve().parent
    _models = sorted((notebook_root / "PyTorchModels").glob("*.pth"))
    _default = "PyTorchModels/pytorch_computer_vision_model_2.pth"
    model_path = mo.ui.text(value=_default, label="Archivo de pesos (.pth)", full_width=True)
    reload_model = mo.ui.run_button(label="Recargar pesos")
    mo.vstack([
        mo.md("**Modelo:** " + (", ".join(p.name for p in _models) or "No se encontraron archivos .pth")),
        mo.hstack([model_path, reload_model]),
    ])
    return model_path, notebook_root, reload_model


@app.cell
def _(load_cnn, mo, model_path, notebook_root, reload_model):
    _reload = reload_model.value
    _path = notebook_root / model_path.value
    mo.stop(not _path.is_file(), mo.md(f"No se encontró el archivo: `{_path}`"))
    try:
        cnn = load_cnn(_path)
    except (RuntimeError, KeyError, ValueError, EOFError, OSError) as _error:
        mo.stop(True, mo.md(f"No se pudo cargar como MNISTModelV2: `{_error}`"))
    mo.md(f"CNN cargada en CPU · **{sum(p.numel() for p in cnn.parameters()):,} parámetros** · modo evaluación.")
    return (cnn,)


@app.cell
def _(anywidget, traitlets):
    class WebcamWidget(anywidget.AnyWidget):
        frame = traitlets.Dict(default_value={}).tag(sync=True)
        _esm = r"""
        export function render({ model, el }) {
            const box = document.createElement("div");
            const video = document.createElement("video");
            video.autoplay = true; video.muted = true; video.playsInline = true;
            video.style.cssText = "width:100%;max-width:480px;display:block;border-radius:8px";
            const controls = document.createElement("div");
            controls.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;margin:10px 0;align-items:center";
            const start = document.createElement("button"); start.textContent = "Activar cámara";
            const snap = document.createElement("button"); snap.textContent = "Capturar cuadro";
            const stop = document.createElement("button"); stop.textContent = "Detener cámara";
            const live = document.createElement("input"); live.type = "checkbox";
            const liveLabel = document.createElement("label");
            liveLabel.append(live, " Actualización continua");
            const rate = document.createElement("select");
            for (const fps of [0.5, 1, 2, 4]) {
                const option = document.createElement("option");
                option.value = fps; option.textContent = `${fps} cuadros/s`;
                rate.append(option);
            }
            rate.value = "1";
            const status = document.createElement("p");
            status.textContent = "La cámara se activa únicamente al pulsar el botón.";
            snap.disabled = stop.disabled = true;
            let stream = null, interval = null, seq = 0, disposed = false;
            const canvas = document.createElement("canvas");
            function capture() {
                if (!stream || !video.videoWidth || document.hidden) return;
                // Limita el tamaño enviado a Python; la inferencia utiliza 28×28.
                const factor = Math.min(1, 640 / video.videoWidth);
                canvas.width = Math.round(video.videoWidth * factor);
                canvas.height = Math.round(video.videoHeight * factor);
                canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
                model.set("frame", {data: canvas.toDataURL("image/jpeg", 0.8), seq: ++seq});
                model.save_changes();
                status.textContent = `Cuadro ${seq} enviado · ${canvas.width}×${canvas.height}`;
            }
            function schedule() {
                clearInterval(interval); interval = null;
                if (stream && live.checked) interval = setInterval(capture, 1000 / Number(rate.value));
            }
            function shutdown() {
                clearInterval(interval); interval = null;
                if (stream) stream.getTracks().forEach(track => track.stop());
                stream = null; video.srcObject = null;
                snap.disabled = stop.disabled = true; start.disabled = false;
                status.textContent = "Cámara detenida. Se conserva la última captura.";
            }
            start.onclick = async () => {
                start.disabled = true;
                try {
                    if (!navigator.mediaDevices?.getUserMedia) {
                        throw new Error("La cámara requiere localhost o HTTPS y un navegador compatible.");
                    }
                    const acquired = await navigator.mediaDevices.getUserMedia({
                        video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false
                    });
                    if (disposed) { acquired.getTracks().forEach(t => t.stop()); return; }
                    stream = acquired; video.srcObject = stream;
                    await video.play(); snap.disabled = stop.disabled = false;
                    status.textContent = "Cámara activa. Captura un cuadro o activa la actualización continua.";
                    schedule();
                } catch (error) {
                    shutdown(); status.textContent = `No se pudo activar: ${error.message}`;
                }
            };
            snap.onclick = capture; stop.onclick = shutdown;
            live.onchange = schedule; rate.onchange = schedule;
            controls.append(start, snap, stop, liveLabel, rate);
            box.append(video, controls, status); el.append(box);
            return () => { disposed = true; shutdown(); };
        }
        """

    return (WebcamWidget,)


@app.cell
def _(WebcamWidget, mo):
    camera = mo.ui.anywidget(WebcamWidget())
    return (camera,)


@app.cell
def _(mo):
    source = mo.ui.dropdown(
        options=["MNIST local", "Webcam", "Archivo"], value="MNIST local", label="Fuente",
    )
    sample_index = mo.ui.slider(start=0, stop=9999, value=0, label="Índice de prueba MNIST", show_value=True)
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"], multiple=False, label="Subir imagen")
    mo.hstack([source, sample_index, upload], wrap=True)
    return sample_index, source, upload


@app.cell
def _(camera, mo, source):
    mo.stop(source.value != "Webcam")
    mo.vstack([
        camera,
        mo.md("Permite el acceso en el navegador. Empieza con **1 cuadro/s**; reduce la frecuencia si las salidas se retrasan. La cámara sigue activa hasta detenerla, aunque cambies de fuente."),
    ])
    return


@app.cell
def _(mo):
    crop = mo.ui.slider(start=10, stop=100, step=5, value=100, show_value=True, label="Recorte central (%)")
    invert = mo.ui.checkbox(value=True, label="Invertir colores (papel blanco)")
    contrast = mo.ui.checkbox(value=True, label="Ajustar contraste")
    center = mo.ui.checkbox(value=True, label="Centrar dígito y dejar margen")
    binary = mo.ui.checkbox(value=False, label="Binarizar")
    threshold = mo.ui.slider(start=0, stop=255, value=128, show_value=True, label="Umbral del trazo claro")
    mo.vstack([
        mo.md("**Preparación de webcam/archivo.** MNIST conserva su imagen original. El recorte utiliza un cuadrado en el centro; reduce su tamaño y coloca ahí el número. El centrado detecta píxeles claros, por lo que sombras o elementos del fondo pueden interferir."),
        mo.hstack([crop, invert, contrast, center], wrap=True),
        mo.hstack([binary, threshold]),
    ])
    return binary, center, contrast, crop, invert, threshold


@app.cell
def _(cnn, mo):
    _options = {}
    for _block_name in ("block_1", "block_2"):
        for _index, _layer in enumerate(getattr(cnn, _block_name)):
            _key = f"{_block_name}.{_index}"
            _options[f"{_key} · {_layer.__class__.__name__}"] = _key
    layer = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Capa observada")
    scale = mo.ui.slider(start=0.1, stop=10, step=0.1, value=1, show_value=True, label="Límite de color")
    columns = mo.ui.slider(start=2, stop=10, value=5, show_value=True, label="Columnas del grid")
    mo.hstack([layer, scale, columns], wrap=True)
    return columns, layer, scale


@app.cell
def _(
    Image,
    base64,
    camera,
    io,
    mnist_sample,
    mo,
    notebook_root,
    sample_index,
    source,
    upload,
):
    true_label = None
    if source.value == "MNIST local":
        try:
            raw_image, true_label = mnist_sample(notebook_root, sample_index.value)
        except (OSError, ValueError) as _error:
            mo.stop(True, mo.md(f"MNIST local no está disponible: `{_error}`. Selecciona Webcam o Archivo."))
        frame_label = f"MNIST #{sample_index.value} · etiqueta {true_label}"
    elif source.value == "Webcam":
        _frame = camera.frame
        mo.stop(not _frame, mo.md("Activa la cámara y pulsa **Capturar cuadro** o **Actualización continua**."))
        raw_image = Image.open(io.BytesIO(base64.b64decode(_frame["data"].split(",", 1)[1]))).convert("RGB")
        frame_label = f"Webcam · cuadro {_frame['seq']}"
    else:
        mo.stop(not upload.value, mo.md("Sube una imagen para continuar."))
        try:
            raw_image = Image.open(io.BytesIO(upload.value[0].contents)).convert("RGB")
        except (OSError, ValueError) as _error:
            mo.stop(True, mo.md(f"No se pudo abrir la imagen: `{_error}`"))
        frame_label = "Archivo subido"
    return frame_label, raw_image, true_label


@app.cell
def _(
    binary,
    center,
    contrast,
    crop,
    invert,
    prepare_image,
    raw_image,
    source,
    threshold,
):
    if source.value == "MNIST local":
        cropped_image, input_image, input_tensor = prepare_image(raw_image)
    else:
        cropped_image, input_image, input_tensor = prepare_image(
            raw_image, crop.value, invert.value, contrast.value,
            threshold.value if binary.value else None, center.value,
        )
    return cropped_image, input_image, input_tensor


@app.cell
def _(cnn, extract_activations, input_tensor):
    activations, probabilities = extract_activations(cnn, input_tensor)
    return activations, probabilities


@app.cell
def _(cropped_image, frame_label, image_uri, input_image, mo, raw_image):
    _panels = []
    for _title, _image in [
        ("Captura original", raw_image), ("Recorte y ajuste", cropped_image),
        ("Entrada de la CNN · 1×1×28×28", input_image),
    ]:
        _panels.append(
            f'<figure style="margin:0;text-align:center"><figcaption>{_title}</figcaption>'
            f'<img src="{image_uri(_image)}" style="height:180px;max-width:100%;object-fit:contain;image-rendering:pixelated" /></figure>'
        )
    mo.vstack([
        mo.md(f"**{frame_label}**"),
        mo.Html('<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px">' + ''.join(_panels) + '</div>'),
    ])
    return


@app.cell
def _(activation_grid, activations, columns, layer, mo, scale):
    selected_maps = activations[layer.value]
    # Las convoluciones producen valores positivos y negativos; ReLU/pooling no.
    _signed = layer.value.endswith(".0") or layer.value.endswith(".2")
    _range = f"−{scale.value:g} a +{scale.value:g}" if _signed else f"0 a {scale.value:g}"
    mo.vstack([
        mo.md(f"### {layer.value} · salida `{tuple(selected_maps.shape)}`\n\nEscala fija: **{_range}**. "
              "Los valores fuera de escala se saturan en el color extremo. Ajusta el límite si muchos píxeles quedan saturados."),
        mo.Html(activation_grid(selected_maps, scale.value, columns.value, _signed)),
    ])
    return


@app.cell
def _(mo, probabilities, true_label):
    _prediction = int(probabilities.argmax())
    _confidence = float(probabilities[_prediction])
    _truth = f" · etiqueta real **{true_label}**" if true_label is not None else ""
    _bars = []
    for _digit, _prob in enumerate(probabilities.tolist()):
        _bars.append(
            '<div style="display:flex;gap:8px;align-items:center;margin:3px 0">'
            f'<span style="width:18px">{_digit}</span>'
            f'<progress value="{_prob}" max="1" style="width:200px"></progress>'
            f'<span>{_prob:.1%}</span></div>'
        )
    mo.vstack([
        mo.md(f"### Predicción: **{_prediction}** · softmax **{_confidence:.1%}**{_truth}"),
        mo.Html(''.join(_bars)),
        mo.md("La CNN siempre elige una de sus diez clases, incluso si la captura no contiene un dígito. "
              "El porcentaje softmax no garantiza que la predicción sea correcta."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Preguntas para explorar

    - ¿Qué canales responden al trazo y cuáles al fondo?
    - Compara una convolución con su ReLU: ¿qué sucede con los valores negativos?
    - Observa `block_1.4` y `block_2.4`: las dimensiones pasan de 28×28 a 14×14 y 7×7.
    - Desplaza o gira un número. ¿Cambian los mapas y la predicción?
    - Compara MNIST con una foto del mismo dígito. ¿Cómo influye la preparación?

    Un canal combina información de los canales de entrada mediante pesos aprendidos.
    Los mapas muestran esa respuesta, **no son imágenes de los kernels**.
    La escala permanece fija entre cuadros de una misma capa. Cambiarla altera la
    visualización, pero no las activaciones ni la predicción.
    """)
    return


if __name__ == "__main__":
    app.run()
