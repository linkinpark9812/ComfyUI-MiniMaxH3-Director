"""WAV serialization and synchronized H.264 exports; no sampler/model mutation."""

from contextlib import ExitStack
from fractions import Fraction
from functools import reduce
import io
import math
import os
from pathlib import Path
from uuid import uuid4
import wave

import av
import numpy as np
from PIL import Image


def encode_wav(samples, sample_rate, duration):
    """Serialize [channels, samples] floats as PCM16, aligned to the shot duration."""
    samples = np.asarray(samples, dtype=np.float32)
    sample_rate = int(sample_rate)
    if samples.ndim != 2 or samples.shape[0] not in (1, 2) or samples.shape[1] == 0:
        raise ValueError("Audio preview requires nonempty mono or stereo samples")
    if sample_rate <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid audio sample rate or duration")
    if not np.isfinite(samples).all():
        raise ValueError("Audio preview contains non-finite samples")
    length = max(1, round(duration * sample_rate))
    # H3 rounds its audio latent to 40 Hz; trim/pad the decoded tail to video time.
    samples = samples[:, :length]
    if samples.shape[1] < length:
        samples = np.pad(samples, ((0, 0), (0, length - samples.shape[1])))
    pcm = (np.clip(samples, -1, 1).T * 32767).round().astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(samples.shape[0])
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return output.getvalue(), {"sample_rate": sample_rate, "channels": samples.shape[0],
                               "samples": length, "duration": length / sample_rate}


def build_av_preview(webp_path, wav_path, destination):
    """Build a lightweight playback copy for older runs without a saved MP4."""
    return _build_av_mp4(webp_path, wav_path, destination, max_resolution=768,
                         crf=23, preset="ultrafast", audio_bitrate=128000)


def build_av_export(webp_path, wav_path, destination, *, frames=None):
    """Full preview resolution, high-quality H.264/AAC; prefer pre-WebP frames."""
    return _build_av_mp4(webp_path, wav_path, destination, max_resolution=0,
                         crf=21, preset="slow", audio_bitrate=320000, frames=frames)


def _build_av_mp4(webp_path, wav_path, destination, *, max_resolution,
                  crf, preset, audio_bitrate, frames=None):
    """Mux images and WAV on one real-time clock, publishing only a complete MP4.

    WebP durations may include merged identical frames. Preserve their relative timing
    while matching the entire image sequence to the WAV's real duration. Original
    captured frames have uniform timing and avoid an extra lossy WebP decode.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid4().hex + ".part")
    with wave.open(str(wav_path), "rb") as source:
        channels, rate, count = source.getnchannels(), source.getframerate(), source.getnframes()
        if source.getsampwidth() != 2 or channels not in (1, 2) or count <= 0:
            raise ValueError("Unsupported preview WAV format")
        samples = np.frombuffer(source.readframes(count), dtype="<i2").reshape(-1, channels)
        samples = np.ascontiguousarray(samples.T, dtype=np.float32) / 32768
    try:
        with ExitStack() as stack:
            if frames is None:
                source = stack.enter_context(Image.open(webp_path))
                durations = []
                for index in range(source.n_frames):
                    source.seek(index)
                    source.load()
                    durations.append(max(1, int(source.info.get("duration", 100))))
                source_size = source.size
            else:
                if not frames or any(frame.size != frames[0].size for frame in frames):
                    raise ValueError("MP4 requires nonempty frames of equal size")
                durations = [1] * len(frames)
                source_size = frames[0].size
            tick_ms = reduce(math.gcd, durations)
            ticks = sum(durations) // tick_ms
            fps = Fraction(ticks * rate, count)
            scale = min(1.0, max_resolution / max(source_size)) if max_resolution else 1.0
            # H.264 yuv420p needs even dimensions. Pad exports, preserving all pixels.
            size = (tuple(max(2, int(edge * scale) // 2 * 2) for edge in source_size)
                    if max_resolution else tuple((edge + 1) // 2 * 2 for edge in source_size))
            with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart"}) as output:
                video = output.add_stream("libx264", rate=fps)
                video.width, video.height = size
                video.pix_fmt = "yuv420p"
                video.options = {"crf": str(crf), "preset": preset}
                audio = output.add_stream("aac", rate=rate)
                audio.layout = "stereo" if channels == 2 else "mono"
                audio.bit_rate = audio_bitrate
                # Feed audio alongside video so the muxer need not buffer the whole clip.
                audio_cursor = 0
                tick = 0
                for index, duration_ms in enumerate(durations):
                    if frames is None:
                        source.seek(index)
                        image = source.convert("RGB")
                    else:
                        image = frames[index].convert("RGB")
                    if image.size != size:
                        if max_resolution:
                            image = image.resize(size, Image.Resampling.LANCZOS)
                        else:
                            padded = Image.new("RGB", size)
                            padded.paste(image, (0, 0))
                            image = padded
                    for _ in range(duration_ms // tick_ms):
                        frame = av.VideoFrame.from_image(image)
                        frame.pts, frame.time_base = tick, 1 / fps
                        for packet in video.encode(frame):
                            output.mux(packet)
                        tick += 1
                        end = min(count, round(tick * count / ticks))
                        while audio_cursor < end:
                            chunk_end = min(end, audio_cursor + 1024)
                            frame = av.AudioFrame.from_ndarray(
                                np.ascontiguousarray(samples[:, audio_cursor:chunk_end]),
                                format="fltp", layout=audio.layout.name)
                            frame.sample_rate = rate
                            frame.pts, frame.time_base = audio_cursor, Fraction(1, rate)
                            for packet in audio.encode(frame):
                                output.mux(packet)
                            audio_cursor = chunk_end
                for stream in (video, audio):
                    for packet in stream.encode(None):
                        output.mux(packet)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
