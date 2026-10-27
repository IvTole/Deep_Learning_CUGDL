"""Inferencia y visualización de SSDLite; sin entrenamiento ni captura de cámara en Python."""

import base64
import io
from time import perf_counter

import anywidget
import numpy as np
import torch
import traitlets
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)


# Bloques completos: sus salidas no se alteran después por operaciones in-place.
LAYER_OPTIONS = {
    "Inicial · 16 canales · 160×160": ("backbone.features.0.0", 16),
    "Intermedia 1 · 24 canales · 80×80": ("backbone.features.0.3", 24),
    "Intermedia 2 · 40 canales · 40×40": ("backbone.features.0.6", 40),
    "Profunda · 480 canales · 10×10": ("backbone.features.1.3", 480),
}


def load_detector(device="cpu"):
    """Descarga oficial en el caché de Torch Hub solo si los pesos no existen."""
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights, progress=True)
    return model.to(device).eval(), weights


def prepare_frame(image, max_side=640):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def infer_frame(model, weights, image, mode="both", layer_name="backbone.features.0.0",
                first_channel=0, channel_count=8):
    """Una pasada; copia solo canales solicitados y retira siempre el hook.

    mode: detections, both o features. features ejecuta el prefijo y retorna temprano.
    Las cajas de salida usan las dimensiones de image, no las internas de 320×320.
    """
    if mode not in {"detections", "both", "features"}:
        raise ValueError("Modo desconocido")
    if layer_name not in {value[0] for value in LAYER_OPTIONS.values()}:
        raise ValueError("Capa no disponible para la salida temprana")
    if first_channel < 0 or channel_count < 1:
        raise ValueError("Selecciona canales válidos")
    device = next(model.parameters()).device
    tensor = weights.transforms()(image.convert("RGB")).to(device)
    captured = {}
    hook = None

    def capture(_module, _inputs, output):
        if first_channel >= output.shape[1]:
            raise ValueError("El canal inicial excede los canales de la capa")
        captured["shape"] = tuple(output.shape)
        # clone evita cambios por operaciones in-place; no retenemos todo el tensor.
        captured["maps"] = output[0, first_channel:first_channel + channel_count].detach().clone().cpu()

    model.eval()
    _synchronize(device)
    started = perf_counter()
    try:
        if mode == "both":
            hook = model.get_submodule(layer_name).register_forward_hook(capture)
        with torch.inference_mode():
            if mode == "features":
                # Reutiliza exactamente la normalización y resize internos del detector.
                transformed, _ = model.transform([tensor], None)
                value = transformed.tensors
                found = False
                for stage_index, stage in enumerate(model.backbone.features):
                    for block_index, block in enumerate(stage):
                        value = block(value)
                        if f"backbone.features.{stage_index}.{block_index}" == layer_name:
                            capture(block, (), value)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise ValueError("La estructura del backbone cambió; revisa LAYER_OPTIONS")
                output = None
            else:
                output = {key: value.detach().cpu() for key, value in model([tensor])[0].items()}
        _synchronize(device)
        elapsed_ms = (perf_counter() - started) * 1000
    finally:
        if hook is not None:
            hook.remove()
    return {"detections": output, "maps": captured.get("maps"),
            "shape": captured.get("shape"), "elapsed_ms": elapsed_ms,
            "first_channel": first_channel}


def filter_detections(prediction, threshold=0.5, class_id=None, max_boxes=15):
    """Filtra salidas ya procesadas por NMS, sin repetir la inferencia."""
    keep = prediction["scores"] >= threshold
    if class_id is not None:
        keep &= prediction["labels"] == class_id
    indices = torch.where(keep)[0]
    indices = indices[torch.argsort(prediction["scores"][indices], descending=True)][:max_boxes]
    return {key: value[indices] for key, value in prediction.items()}


def annotate_detections(image, detections, categories):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    rows = []
    for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
        coordinates = tuple(float(value) for value in box)
        name, probability = categories[int(label)], float(score)
        color = (20, 150, 80) if int(label) == 1 else (210, 85, 15)
        draw.rectangle(coordinates, outline=color, width=3)
        text = f"{name} {probability:.0%}"
        x, y = max(0, coordinates[0]), max(0, coordinates[1] - 20)
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x, y), text, fill="white", font=font)
        rows.append({"clase COCO": name, "score": round(probability, 4),
                     "xyxy (px)": str(tuple(round(value, 1) for value in coordinates))})
    return canvas, rows


def image_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def activation_grid(maps, first_channel=0, limit=3.0, columns=4):
    from matplotlib import colormaps

    tiles = []
    for index, channel in enumerate(maps):
        array = channel.numpy()
        normalized = np.clip((array / limit + 1) / 2, 0, 1)
        rgb = (colormaps["coolwarm"](normalized)[..., :3] * 255).astype(np.uint8)
        picture = Image.fromarray(rgb).resize((140, 140), Image.Resampling.NEAREST)
        saturated = float(np.mean(np.abs(array) > limit))
        tiles.append(
            '<figure style="margin:0;text-align:center">'
            f'<img src="{image_uri(picture)}" style="width:100%;max-width:140px" />'
            f'<figcaption>Canal {first_channel + index}<br><small>'
            f'mín {array.min():.2f} · máx {array.max():.2f}<br>'
            f'{saturated:.0%} fuera de escala</small></figcaption></figure>'
        )
    return (f'<div style="display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));'
            'gap:12px">' + ''.join(tiles) + '</div>')


class DetectionWebcamWidget(anywidget.AnyWidget):
    """Una captura pendiente como máximo; Python confirma al terminar la inferencia."""
    frame = traitlets.Dict(default_value={}).tag(sync=True)
    processed_seq = traitlets.Int(0).tag(sync=True)
    enabled = traitlets.Bool(False).tag(sync=True)
    _esm = r"""
    export function render({ model, el }) {
        const video = document.createElement("video");
        video.autoplay = true; video.muted = true; video.playsInline = true;
        video.style.cssText = "width:100%;max-width:480px;display:block;border-radius:8px";
        const controls = document.createElement("div");
        controls.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0";
        const start = document.createElement("button"); start.textContent = "Activar cámara";
        const snap = document.createElement("button"); snap.textContent = "Capturar cuadro";
        const stop = document.createElement("button"); stop.textContent = "Detener cámara";
        const live = document.createElement("input"); live.type = "checkbox";
        const liveLabel = document.createElement("label"); liveLabel.append(live, " Actualización continua");
        const rate = document.createElement("select");
        for (const fps of [0.5, 1, 2, 4]) {
            const option = document.createElement("option");
            option.value = fps; option.textContent = `${fps} capturas/s como máximo`; rate.append(option);
        }
        rate.value = "1";
        const status = document.createElement("p");
        status.textContent = "La cámara solo se activa al pulsar el botón.";
        const canvas = document.createElement("canvas");
        let stream = null, timer = null, disposed = false;
        let seq = model.get("frame").seq || 0;
        snap.disabled = stop.disabled = true;
        function capture() {
            if (!stream || !video.videoWidth || document.hidden || !model.get("enabled")) return;
            if (seq > model.get("processed_seq")) {
                status.textContent = `Esperando a Python · cuadro ${seq}. No se acumulan capturas.`;
                return;
            }
            const factor = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
            canvas.width = Math.max(1, Math.round(video.videoWidth * factor));
            canvas.height = Math.max(1, Math.round(video.videoHeight * factor));
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            model.set("frame", {data: canvas.toDataURL("image/jpeg", 0.8), seq: ++seq});
            model.save_changes();
            status.textContent = `Cuadro ${seq} enviado a Python · ${canvas.width}×${canvas.height}`;
        }
        function schedule() {
            clearInterval(timer); timer = null;
            if (stream && live.checked) timer = setInterval(capture, 1000 / Number(rate.value));
        }
        function shutdown() {
            clearInterval(timer); timer = null;
            if (stream) stream.getTracks().forEach(track => track.stop());
            stream = null; video.srcObject = null;
            start.disabled = false; snap.disabled = stop.disabled = true;
            status.textContent = "Cámara detenida. Se conserva la última captura.";
        }
        start.onclick = async () => {
            start.disabled = true;
            try {
                if (!navigator.mediaDevices?.getUserMedia) throw new Error("Usa localhost o HTTPS.");
                const acquired = await navigator.mediaDevices.getUserMedia({
                    video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false
                });
                if (disposed || !model.get("enabled")) {
                    acquired.getTracks().forEach(track => track.stop()); start.disabled = false; return;
                }
                stream = acquired; video.srcObject = stream;
                await video.play(); snap.disabled = stop.disabled = false;
                status.textContent = "Cámara activa. Captura un cuadro para comenzar.";
                schedule();
            } catch (error) {
                shutdown(); status.textContent = `No se pudo activar: ${error.message}`;
            }
        };
        snap.onclick = capture; stop.onclick = shutdown;
        live.onchange = schedule; rate.onchange = schedule;
        const onEnabled = () => { if (!model.get("enabled")) shutdown(); };
        model.on("change:enabled", onEnabled);
        const onProcessed = () => {
            if (stream) status.textContent = `Python terminó el cuadro ${model.get("processed_seq")}.`;
        };
        model.on("change:processed_seq", onProcessed);
        controls.append(start, snap, stop, liveLabel, rate);
        el.append(video, controls, status);
        return () => {
            disposed = true; shutdown();
            model.off("change:enabled", onEnabled); model.off("change:processed_seq", onProcessed);
        };
    }
    """
