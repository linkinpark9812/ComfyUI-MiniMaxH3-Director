"""Live sampling preview for MiniMax H3.

Why this exists
---------------
ComfyUI core does ship `latent_rgb_factors` for MiniMaxH3Video, so previews are not
missing — but `Latent2RGBPreviewer` renders `x0[0, :, 0]`, i.e. the first latent frame
only. You watch a single still while a five-second shot is being sampled.

KJNodes' Preview Override does the good version of this (in-node preview, optional full
VAE decode), but its video paths are gated on `_is_ltx_latent_format` /
`_is_ltx2_diffusion_model`, and nothing there unpacks H3's packed AV latent — so on
MiniMax it falls through to the same single frame. This node is the H3 equivalent.

The one non-obvious detail
--------------------------
`CFGGuider.sample` packs the video and audio streams into ONE flat tensor and only then
wraps the callback with the nested view. That wrapper sits *behind* an OUTER_SAMPLE
wrapper in the call chain, so what reaches this callback is the flat pack, not the
NestedTensor — `_video_stream` unpacks it with core's own `unpack_latents`.
"""

import base64
import io as _io
import logging
import struct
import time

import torch
import torch.nn.functional as F
from PIL import Image

import comfy.patcher_extension
from comfy.model_management import InterruptProcessingException
from comfy_execution.utils import get_executing_context
import folder_paths
import comfy.utils
import latent_preview
import server
from comfy_api.latest import io
from protocol import BinaryEventTypes

from .minimax_preview_archive import DEFAULT_FOLDER, PreviewArchiveSession, normalize_folder, new_run_id
from .minimax_core import audio_nodes

log = logging.getLogger(__name__)

EVENT = "minimax_h3_preview"
ARCHIVE_EVENT = "minimax_h3_preview_archive"
DECODE_FAST = "latent2rgb (fast)"
DECODE_VAE = "vae (quality)"
PLAYBACK_TRUE = "true speed"
PLAYBACK_SOURCE = "source fps"
MODEL_FPS = 24.0            # H3's native output rate
TARGET_NODE = "node"
TARGET_SAMPLER = "sampler (VHS)"
TARGET_BOTH = "both"


def _video_stream(x0, latent_shapes=None):
    """Pull the [B, C, T, h, w] video latent out of whatever the sampler handed us."""
    if x0 is None:
        return None
    if getattr(x0, "is_nested", False):
        return x0.tensors[0]
    if x0.ndim == 5:
        return x0
    if latent_shapes and len(latent_shapes) > 1:
        return comfy.utils.unpack_latents(x0, list(latent_shapes))[0]
    return None


def _audio_stream(x0, latent_shapes=None):
    """Read H3's [B, 32, stereo, T] audio stream without modifying the sampler pack."""
    if x0 is None:
        return None
    if getattr(x0, "is_nested", False):
        streams = x0.tensors
    elif latent_shapes and len(latent_shapes) > 1:
        streams = comfy.utils.unpack_latents(x0, list(latent_shapes))
    else:
        return None
    return streams[1] if len(streams) > 1 else None


def _decode_audio_wav(vae, x0, duration, model, latent_shapes=None):
    from .minimax_preview_audio import encode_wav

    if x0 is None:
        raise ValueError("No H3 audio latent is available for this step")
    # Callback x0 is still in sampler space. H3 scales the packed audio onto the
    # video schedule; the final sampler output undoes this in process_latent_out.
    # Apply that same model conversion BEFORE extracting audio, since it needs the
    # full AV layout. Use private float32 storage so neither conversion nor VAE
    # decoding can alter the sampler's callback tensors. NestedTensor has no detach.
    source = x0 if getattr(x0, "is_nested", False) else x0.detach()
    source = source.to(dtype=torch.float32, copy=True)
    latent = _audio_stream(model.process_latent_out(source), latent_shapes)
    if latent is None or latent.ndim != 4:
        raise ValueError("No H3 audio latent is available for this step")
    # Use ComfyUI's own audio decoding/normalization and never lend it sampler storage.
    decoded = audio_nodes().VAEDecodeAudio.execute(
        vae=vae, samples={"samples": latent[:1].detach().clone()}).args[0]
    samples = decoded["waveform"][0].detach().float().cpu().numpy()
    return encode_wav(samples, decoded["sample_rate"], duration)


def _pick_frames(video, max_frames):
    """Evenly thin [B, C, T, h, w] down to at most max_frames along T."""
    t = video.shape[2]
    if max_frames <= 0 or t <= max_frames:
        return video
    idx = torch.linspace(0, t - 1, max_frames).round().long().unique()
    return video[:, :, idx]


def pixel_frames_from_latent_t(latent_t):
    """How many output frames one H3 video latent covers.

    The latent is compressed ~3.35x in time (core's `video_latent_t`: 17k+5 pixel frames
    become 5k+2 latent frames), so a preview that plays latent frames at the video's fps
    runs more than three times too fast. Inverting that mapping is what keeps the preview
    honest about the shot's real speed.
    """
    if latent_t <= 2:
        return 5
    k, remainder = divmod(int(latent_t) - 2, 5)
    if remainder == 0:
        return 17 * k + 5
    return max(1, int(round(latent_t * 17.0 / 5.0)))   # off-grid latent: approximate


class _RGBFactors:
    def __init__(self, latent_format):
        factors = getattr(latent_format, "latent_rgb_factors", None)
        if factors is None:
            raise ValueError("latent format has no latent_rgb_factors")
        # stored as an F.linear weight: [out=3, in=C] -> channel count is shape[1]
        self.w = torch.tensor(factors, device="cpu").transpose(0, 1)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        self.b = torch.tensor(bias, device="cpu") if bias is not None else None

    def __call__(self, video):
        """[B, C, T, h, w] -> [N, h, w, 3] in 0..1"""
        chans = self.w.shape[1]
        moved = video.movedim(2, 1)                       # [B, T, C, h, w]
        # flatten batch-major; take the shape AFTER the movedim, not the caller's
        x = moved.reshape((-1,) + tuple(moved.shape[-3:]))[:, :chans].to(torch.float32)
        w = self.w.to(dtype=x.dtype, device=x.device)
        b = self.b.to(dtype=x.dtype, device=x.device) if self.b is not None else None
        return ((F.linear(x.movedim(1, -1), w, bias=b) + 1.0) / 2.0).clamp(0, 1)


def _vae_decode(vae, video):
    """Full-quality decode of [B, C, T, h, w] -> [N, h, w, 3] in 0..1."""
    images = vae.decode(video)
    if images.ndim == 5:                       # [B, T, h, w, 3]
        images = images.reshape(-1, *images.shape[-3:])
    return images.clamp(0, 1).to(torch.float32).cpu()


def _to_pil(images, max_res):
    """Render frames at `max_res` on the long edge — a target, not just a ceiling.

    latent2rgb frames arrive at latent resolution (a 1344x768 shot is an 84x48 grid), so
    without an upscale the preview is a postage stamp; with a nearest-neighbour upscale it
    is a mosaic. Smooth interpolation reads as "approximate", which is what it is — switch
    decode to 'vae (quality)' for real detail.
    """
    out = []
    for frame in images:
        arr = (frame * 255.0).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(arr)
        if max_res > 0:
            longest = max(img.width, img.height)
            if longest != max_res and longest > 0:
                scale = max_res / float(longest)
                size = (max(1, int(round(img.width * scale))),
                        max(1, int(round(img.height * scale))))
                img = img.resize(size, Image.LANCZOS if scale < 1.0 else Image.BICUBIC)
        out.append(img)
    return out


def _encode_animated_webp(frames, fps, quality):
    if not frames:
        return None
    buf = _io.BytesIO()
    try:
        frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:],
                       duration=max(1, int(round(1000 / max(1, fps)))), loop=0,
                       quality=quality, method=0)
    except InterruptProcessingException:
        raise
    except Exception as e:
        log.warning("[MiniMaxDirector] animated WebP encode failed: %s", e)
        return None
    return buf.getvalue()


def throttle_gap(cost_seconds, max_overhead_percent):
    """Space live display updates by cost; archive capture cadence is independent."""
    if max_overhead_percent <= 0 or cost_seconds <= 0:
        return 0.0
    return cost_seconds * (100.0 / float(max_overhead_percent) - 1.0)


class _VHSStreamer:
    """Streams individual frames to VideoHelperSuite's animated latent-preview player."""

    def __init__(self, rate, client_id, node_id):
        self.client_id = client_id
        self.node_id = str(node_id or "")
        self.rate = max(1, int(rate))
        self.first = True
        self.last_time = 0.0
        self.cursor = 0

    def send(self, pil_frames, rate=None):
        srv = server.PromptServer.instance
        if self.client_id is None:
            return 0
        total = len(pil_frames)
        if total == 0:
            return 0
        if rate and self.first:
            # locked in with the handshake — the player is told the rate exactly once
            self.rate = max(1, int(round(rate)))
        now = time.time()
        count = int((now - self.last_time) * self.rate)
        self.last_time += count / self.rate
        if count > total:
            count = total
        elif count <= 0:
            return 0
        if self.first:
            self.first = False
            srv.send_sync("VHS_latentpreview",
                          {"length": total, "rate": self.rate, "id": self.node_id}, self.client_id)
            self.last_time = now + 1.0 / self.rate

        order = [(self.cursor + i) % total for i in range(count)]
        node_id = self.node_id.encode("ascii", errors="replace")
        for i in order:
            message = _io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(i.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", node_id))
            pil_frames[i].save(message, format="JPEG", quality=95)
            srv.send_sync(BinaryEventTypes.PREVIEW_IMAGE, message.getvalue(), self.client_id)
        self.cursor = (self.cursor + count) % total
        return count


class _OuterSampleWrapper:
    def __init__(self, node_id, decode_mode, vae, max_resolution, preview_frames,
                 preview_fps, webp_quality, every_n_steps, suppress_default, target,
                 max_overhead=25, playback=None, archive_folder_name=DEFAULT_FOLDER,
                 audio_vae=None):
        self.node_id = node_id
        self.decode_mode = decode_mode
        self.vae = vae
        self.max_resolution = int(max_resolution)
        self.preview_frames = int(preview_frames)
        self.preview_fps = float(preview_fps)
        self.webp_quality = int(webp_quality)
        self.every_n_steps = max(1, int(every_n_steps))
        self.suppress_default = bool(suppress_default)
        self.target = target
        self.max_overhead = max(0, min(100, int(max_overhead)))
        self.playback = playback or PLAYBACK_TRUE
        self.archive_folder_name = archive_folder_name
        self.audio_vae = audio_vae

    def _rate_for(self, shown, pixel_frames):
        """Frames per second to play `shown` images at.

        Two honest answers, and the node used to pick one for you:

        `true speed` spreads the images across the shot's real duration, so the preview
        lasts as long as the finished clip. With latent2rgb that caps out at
        preview_fps / 3.35 — one image per latent frame, and H3 compresses time by that
        much — which looks like a setting being ignored if nobody says so.

        `source fps` plays them at preview_fps flat. That is what ComfyUI's own preview and
        the other packs do, and it is why they show a round 24: the motion reads at normal
        speed but the clip is over in a third of the time. Useful for judging movement,
        misleading about timing.
        """
        if self.playback == PLAYBACK_SOURCE:
            return max(0.1, self.preview_fps)
        return max(0.1, self.preview_fps * shown / max(1, pixel_frames))

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, **kwargs):
        # Resolve at sampling time, not execute(): the patched model may be cached.
        srv = server.PromptServer.instance
        context = get_executing_context()
        client_id = getattr(srv, "client_id", None)
        identity = {
            "node_id": self.node_id, "run_id": new_run_id(),
            "folder_name": normalize_folder(self.archive_folder_name),
            "prompt_id": getattr(context, "prompt_id", None),
            "sampler_id": getattr(context, "node_id", None),
        }
        sequence = 0

        def emit(kind, payload=None, event=ARCHIVE_EVENT):
            nonlocal sequence
            sequence += 1
            if client_id is None:
                return  # None means broadcast in ComfyUI; headless runs must not broadcast.
            try:
                srv.send_sync(event, {**identity, **(payload or {}), "kind": kind,
                                      "sequence": sequence}, client_id)
            except Exception:
                log.warning("[MiniMaxDirector] preview notification failed", exc_info=True)

        session = None
        storage_failed = False
        try:
            session = PreviewArchiveSession(
                folder_paths.get_output_directory(), self.archive_folder_name, self.node_id,
                {k: identity[k] for k in ("prompt_id", "sampler_id")},
                {"every_n_steps": self.every_n_steps, "decode": self.decode_mode,
                 "preview_frames": self.preview_frames, "preview_fps": self.preview_fps,
                 "max_resolution": self.max_resolution, "webp_quality": self.webp_quality,
                 "playback": self.playback, "preview_target": self.target,
                 "capture_audio": self.audio_vae is not None})
            identity["run_id"] = session.manifest["run_id"]
            emit("started", session.snapshot())
        except Exception as exc:
            storage_failed = True
            log.warning("[MiniMaxDirector] cannot create preview archive: %s", exc)
            emit("started", {"archive_state": "unavailable", "error": str(exc),
                             "sampling_status": "running", "artifacts": []})

        def failure(step, exc, storage=False):
            nonlocal storage_failed
            message = str(exc) or type(exc).__name__
            if session:
                session.manifest["capture_failures"].append({"step": step, "error": message})
                if storage:
                    session.manifest["error"] = "Preview storage failed: " + message
            if storage:
                storage_failed = True
            log.warning("[MiniMaxDirector] preview step %s failed: %s", step, message)
            emit("archive_error", {"error": message, "step": step})

        original_decode = latent_preview.LatentPreviewer.decode_latent_to_preview_image
        status, sampling_error = "failed", None
        try:
            to_node = self.target in (TARGET_NODE, TARGET_BOTH)
            vhs = (_VHSStreamer(self.preview_fps, client_id, identity["sampler_id"])
                   if self.target in (TARGET_SAMPLER, TARGET_BOTH) else None)
            to_rgb = None
            if self.decode_mode == DECODE_FAST or self.vae is None:
                try:
                    to_rgb = _RGBFactors(executor.class_obj.model_patcher.model.latent_format)
                except InterruptProcessingException:
                    raise
                except Exception as exc:
                    failure(0, exc)
            if self.suppress_default:
                latent_preview.LatentPreviewer.decode_latent_to_preview_image = \
                    lambda self_, preview_format, x0: None
            presentation = {"finished": 0.0, "gap": 0.0}
            saved_steps = set()

            def combined(step, x0, x, total_steps):
                number = int(step) + 1
                if session:
                    session.manifest.update(last_step=number, total_steps=int(total_steps))
                eligible = step % self.every_n_steps == 0 or number == total_steps
                if eligible and number not in saved_steps:
                    started = time.monotonic()
                    try:
                        video = _video_stream(x0, kwargs.get("latent_shapes"))
                        if video is None or video.ndim != 5:
                            raise ValueError("No usable H3 video latent for this step")
                        if self.audio_vae is not None:
                            video = video[:1]  # One paired clip, not several batches concatenated.
                        pixel_frames = pixel_frames_from_latent_t(int(video.shape[2]))
                        video = _pick_frames(video, self.preview_frames)
                        images = (_vae_decode(self.vae, video)
                                  if self.decode_mode == DECODE_VAE and self.vae is not None
                                  else to_rgb(video))
                        frames = _to_pil(images, self.max_resolution)
                        rate = self._rate_for(len(frames), pixel_frames)
                        data = _encode_animated_webp(frames, rate, self.webp_quality)
                        if not data:
                            raise ValueError("Animated WebP encoding failed")
                        metadata = {"frames": len(frames), "fps": round(rate, 2),
                                    "source_fps": self.preview_fps, "step": number,
                                    "total_steps": int(total_steps), "mode": self.decode_mode,
                                    "playback": self.playback,
                                    "ms": int((time.monotonic() - started) * 1000)}
                        audio_data, audio_metadata = None, None
                        if self.audio_vae is not None:
                            try:
                                audio_data, audio_metadata = _decode_audio_wav(
                                    self.audio_vae, x0, pixel_frames / MODEL_FPS,
                                    executor.class_obj.inner_model, kwargs.get("latent_shapes"))
                            except InterruptProcessingException:
                                raise
                            except Exception as exc:
                                metadata["audio_error"] = str(exc) or type(exc).__name__
                                failure(number, RuntimeError("Audio preview: " + metadata["audio_error"]))
                        metadata["ms"] = int((time.monotonic() - started) * 1000)
                        record = None
                        if session and not storage_failed:
                            try:
                                record = session.save_step(number, data, metadata,
                                                           audio_data, audio_metadata)
                                saved_steps.add(number)
                                if record.get("audio"):
                                    try:
                                        from .minimax_preview_audio import build_av_export

                                        session.save_mp4(number, lambda webp, wav, destination:
                                            build_av_export(webp, wav, destination, frames=frames))
                                    except InterruptProcessingException:
                                        raise
                                    except Exception as exc:
                                        failure(number, RuntimeError(f"MP4 export: {exc}"))
                                record["ms"] = int((time.monotonic() - started) * 1000)
                                session.persist()
                                emit("step_saved", {"artifact": record})
                            except InterruptProcessingException:
                                raise
                            except Exception as exc:
                                failure(number, exc, storage=True)
                        elif session:
                            session.manifest["capture_failures"].append(
                                {"step": number, "error": "Storage unavailable"})
                        # Capture always runs at the selected interval. Only live display
                        # is throttled to avoid restarting an animation every callback.
                        now = time.monotonic()
                        if now - presentation["finished"] >= presentation["gap"]:
                            if to_node:
                                payload = dict(metadata)
                                if record:
                                    payload["artifact"] = record
                                else:
                                    payload["webp"] = base64.b64encode(data).decode("ascii")
                                emit("display", payload, event=EVENT)
                            if vhs:
                                vhs.send(frames, rate)
                            cost = time.monotonic() - started
                            presentation.update(finished=time.monotonic(), gap=max(
                                throttle_gap(cost, self.max_overhead),
                                (audio_metadata["duration"] if audio_metadata else
                                 len(frames) / max(0.1, rate)) if to_node else 0.0))
                    except InterruptProcessingException:
                        raise
                    except Exception as exc:
                        failure(number, exc)
                if callback is not None:
                    callback(step, x0, x, total_steps)

            result = executor(noise, latent_image, sampler, sigmas, denoise_mask, combined,
                              disable_pbar, seed, **kwargs)
            status = "completed"
            return result
        except InterruptProcessingException:
            status = "interrupted"
            raise
        except BaseException as exc:
            sampling_error = str(exc) or type(exc).__name__
            raise
        finally:
            # Restore first; archive I/O must never delay restoring ComfyUI's previewer.
            latent_preview.LatentPreviewer.decode_latent_to_preview_image = original_decode
            if session:
                emit("finalizing", {"sampling_status": status, "archive_state": "finalizing"})
                try:
                    summary = session.finalize(status, sampling_error)
                except Exception as exc:
                    log.warning("[MiniMaxDirector] archive finalization failed", exc_info=True)
                    summary = session.snapshot()
                    summary.update(sampling_status=status, archive_state="unavailable", error=str(exc))
                emit("finished", summary)
            else:
                emit("finished", {"sampling_status": status, "archive_state": "unavailable",
                                  "error": "Preview archive storage was unavailable"})


class MiniMaxH3PreviewOverride(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PreviewOverrideCS",
            display_name="MiniMax H3 Preview Override",
            category="MiniMax H3",
            description=(
                "Live preview of the whole shot while it denoises, shown on this node. "
                "Core previews only ever draw the first latent frame; MiniMax H3's packed "
                "audio+video latent also has to be unpacked first, which the LTX preview "
                "nodes do not do. Wire between the Director's model output and the sampler."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Model to attach the preview to."),
                io.Vae.Input("vae", optional=True,
                             tooltip="minimax_h3_video_vae. Only needed for decode='vae (quality)'."),
                io.Combo.Input("decode", options=[DECODE_FAST, DECODE_VAE], default=DECODE_FAST,
                               tooltip="latent2rgb is a single matmul — effectively free, rough colours. "
                                       "vae is the real decoder: true colours, but it costs real time per "
                                       "preview, so raise every_n_steps with it."),
                io.Combo.Input("preview_target", options=[TARGET_NODE, TARGET_SAMPLER, TARGET_BOTH],
                               default=TARGET_NODE,
                               tooltip="Where the preview appears: on this node, in the sampler's usual "
                                       "preview slot (needs VideoHelperSuite), or both."),
                io.Int.Input("max_resolution", default=512, min=64, max=2048, step=32,
                             tooltip="Long edge of the preview image. With latent2rgb the source "
                                     "is latent-sized (a 1344x768 shot is an 84x48 grid), so this "
                                     "upscales — smooth, but soft. Use decode='vae (quality)' when "
                                     "you need to judge detail."),
                io.Int.Input("preview_frames", default=24, min=1, max=512, step=1,
                             tooltip="How many frames of the shot to show. Frames are thinned evenly, "
                                     "so this caps the cost without cropping the timeline."),
                io.Float.Input("preview_fps", default=24.0, min=1.0, max=60.0, step=1.0,
                               tooltip="The shot's own frame rate — 24 for H3. FLOAT so the "
                                       "Director's 'fps' output can be wired straight in. "
                                       "Whether the preview actually plays at this rate "
                                       "depends on 'playback'; with 'true speed' it is a "
                                       "ceiling, not a promise."),
                io.Int.Input("webp_quality", default=80, min=1, max=100, step=1, optional=True,
                             tooltip="WebP quality of the animation sent to the node."),
                io.Int.Input("every_n_steps", default=1, min=1, max=50, step=1, optional=True,
                             tooltip="Save step 1, then every N steps, plus the final step. "
                                     "Filenames retain the actual step numbers. Default: every step."),
                io.Int.Input("max_preview_overhead", default=25, min=0, max=100, step=5, optional=True,
                             tooltip="Spaces live display updates using preview cost (0 disables). "
                                     "Does not skip archive captures or cap total decoding time. "
                                     "Raise every_n_steps to reduce capture overhead."),
                io.Boolean.Input("suppress_default_preview", default=True, optional=True,
                                 tooltip="Hide ComfyUI's built-in single-frame preview while this runs."),
                # NEW WIDGETS GO LAST. ComfyUI serialises widget values positionally, so
                # inserting one in the middle shifts every value after it in workflows that
                # were saved before it existed — this one first landed between preview_fps
                # and webp_quality and handed a saved 80 to a combo that has no such option.
                io.Combo.Input("playback", options=[PLAYBACK_TRUE, PLAYBACK_SOURCE],
                               default=PLAYBACK_TRUE, optional=True,
                               tooltip="'true speed' spreads the sampled frames across the "
                                       "shot's real duration, so the preview lasts exactly as "
                                       "long as the finished clip — but with latent2rgb that "
                                       "caps at preview_fps / 3.35, because there is one image "
                                       "per latent frame and H3 compresses time by that much. "
                                       "'source fps' plays them at preview_fps flat, like "
                                       "ComfyUI's own preview: motion reads at normal speed, "
                                       "the clip ends early. Judge timing with the first, "
                                       "movement with the second."),
                io.String.Input("archive_folder_name", default=DEFAULT_FOLDER, optional=True,
                                tooltip="Folder name inside the ZIP and under ComfyUI output. "
                                        "Each run is stored separately. Not a server path."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="Connect the MiniMax H3 audio VAE to save a WAV and H.264 MP4 for every "
                                     "captured step and enable synchronized, real-time AV playback. "
                                     "Adds audio decoding work. Leave disconnected for WebP only."),
            ],
            outputs=[io.Model.Output(tooltip="Model with the preview attached.")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, decode=DECODE_FAST, preview_target=TARGET_NODE, max_resolution=512,
                preview_frames=24, preview_fps=24.0, playback=PLAYBACK_TRUE, webp_quality=80,
                every_n_steps=1, max_preview_overhead=25, suppress_default_preview=True,
                vae=None, archive_folder_name=DEFAULT_FOLDER, audio_vae=None) -> io.NodeOutput:
        if decode == DECODE_VAE and vae is None:
            raise ValueError(
                "MiniMax H3 Preview Override: decode is set to 'vae (quality)' but no VAE is "
                "connected. Wire minimax_h3_video_vae into 'vae', or switch decode back to "
                "'latent2rgb (fast)'."
            )

        m = model.clone()
        wrapper = _OuterSampleWrapper(
            str(cls.hidden.unique_id), decode, vae, max_resolution, preview_frames,
            preview_fps, webp_quality, every_n_steps, suppress_default_preview,
            preview_target, max_preview_overhead, playback, archive_folder_name, audio_vae)

        # Register where the sampler actually looks: CFGGuider reads
        #   get_all_wrappers(OUTER_SAMPLE, self.model_options, is_model_options=True)
        # `clone()` deep-copies model_options, so this cannot leak into the source patcher.
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "minimax_h3_preview", wrapper, m.model_options, is_model_options=True)

        registered = comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, m.model_options,
            is_model_options=True)
        if wrapper not in registered and hasattr(m, "add_wrapper_with_key"):
            # a build that still reads the patcher-side dict; checking first avoids
            # registering twice and firing the preview for every step twice over
            log.info("[MiniMaxDirector] using ModelPatcher-side wrapper registration.")
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   "minimax_h3_preview", wrapper)
        return io.NodeOutput(m)
