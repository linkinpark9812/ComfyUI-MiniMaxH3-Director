"""WAV serialization and disposable AV playback copies; no sampler/model mutation."""

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
    """Mux a lightweight, real-time H.264/AAC preview from the saved pair.

    WebP durations may include merged identical frames. Preserve their relative timing
    while matching the entire image sequence to the WAV's real duration. This derived
    MP4 is only a temp cache; original WebP/WAV downloads are never transcoded.
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
        with Image.open(webp_path) as source:
            durations = []
            for index in range(source.n_frames):
                source.seek(index)
                source.load()
                durations.append(max(1, int(source.info.get("duration", 100))))
            tick_ms = reduce(math.gcd, durations)
            ticks = sum(durations) // tick_ms
            fps = Fraction(ticks * rate, count)
            scale = min(1.0, 768 / max(source.size))
            size = tuple(max(2, int(edge * scale) // 2 * 2) for edge in source.size)
            with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart"}) as output:
                video = output.add_stream("libx264", rate=fps)
                video.width, video.height = size
                video.pix_fmt = "yuv420p"
                video.options = {"crf": "23", "preset": "ultrafast"}
                audio = output.add_stream("aac", rate=rate)
                audio.layout = "stereo" if channels == 2 else "mono"
                # Feed audio alongside video so the muxer need not buffer the whole clip.
                audio_cursor = 0
                tick = 0
                for index, duration_ms in enumerate(durations):
                    source.seek(index)
                    image = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
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
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
