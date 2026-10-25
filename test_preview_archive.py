"""Preview storage, sampler lifecycle, and real HTTP tests without a GPU.

Run: python test_preview_archive.py
Storage tests use only the standard library. Pillow enables animated WebP and stubbed
sampler tests; aiohttp enables HTTP downloads and PyAV/NumPy enable real AV codec tests. GPU decoding is
deliberately stubbed, not represented as real model validation.
"""

import asyncio
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import minimax_preview_archive as archive
try:
    import minimax_preview_audio as preview_audio
except ImportError:
    preview_audio = None

try:
    from PIL import Image
except ImportError:
    Image = None
try:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:
    web = None

HERE = Path(__file__).resolve().parent


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = archive.PreviewArchiveSession(self.temp.name, "MyShot", "7")

    def test_names_and_portable_normalization(self):
        for value in ("", "..", "CON", "con.txt", "LPT1", "LPT¹", "a/b\\c", "trailing. ",
                      "a\x00b", "😀" * 100, "CON." + "a" * 100, "été 東京"):
            with self.subTest(value=value):
                result = archive.normalize_folder(value)
                self.assertEqual(archive.normalize_folder(result), result)
                self.assertLessEqual(len(result.encode("utf-8")), 80)
                self.assertNotIn("/", result)
                self.assertNotIn("\\", result)
        self.assertEqual(archive.normalize_folder("MyShot"), "MyShot")

    def test_actual_numbers_zip_contents_and_idempotence(self):
        for number in (1, 6, 11, 30, 999, 1000):
            self.session.save_step(number, f"bytes {number}".encode(), {})
        result = self.session.finalize("completed")
        self.assertEqual(result, self.session.finalize("interrupted"))
        with ZipFile(self.session.directory / "MyShot.zip") as bundle:
            self.assertEqual(bundle.namelist(), [f"MyShot/Step{n:03d}.webp" for n in (1, 6, 11, 30, 999, 1000)])
            for name in bundle.namelist():
                self.assertEqual(bundle.read(name), (self.session.directory / Path(name).name).read_bytes())
        self.assertEqual(result["archive_state"], "ready")
        self.assertFalse(list(self.session.directory.glob("*.part")))

    def test_runs_with_same_name_never_overwrite(self):
        self.session.save_step(1, b"first", {})
        self.session.finalize("completed")
        other = archive.PreviewArchiveSession(self.temp.name, "MyShot", "7")
        other.save_step(1, b"second", {})
        other.finalize("completed")
        self.assertNotEqual(self.session.directory, other.directory)
        self.assertEqual((self.session.directory / "Step001.webp").read_bytes(), b"first")

    def test_empty_cancel_has_named_directory(self):
        result = self.session.finalize("interrupted")
        self.assertEqual(result["sampling_status"], "interrupted")
        with ZipFile(self.session.directory / "MyShot.zip") as bundle:
            self.assertEqual(bundle.namelist(), ["MyShot/"])

    def test_traversal_and_unlisted_steps(self):
        run_id = self.session.manifest["run_id"]
        for folder, run in (("../outside", run_id), ("MyShot", "../../outside"), ("/absolute", run_id)):
            with self.assertRaises(ValueError):
                archive.run_directory(self.temp.name, folder, run)
        with self.assertRaises(FileNotFoundError):
            archive.artifact_path(self.temp.name, "MyShot", run_id, self.session.snapshot(), 1)
        with self.assertRaises(RuntimeError):
            archive.artifact_path(self.temp.name, "MyShot", run_id, self.session.snapshot())

    def test_symlink_escape(self):
        target = Path(self.temp.name) / "outside"
        target.mkdir()
        link = self.session.directory / "Step001.webp"
        outside = target / "secret.webp"
        outside.write_bytes(b"secret")
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Symlink creation unavailable to this account")
        with self.assertRaises(ValueError):
            self.session.save_step(1, b"preview", {})
        self.assertEqual(outside.read_bytes(), b"secret")

    def test_atomic_failure_does_not_publish_partial_file(self):
        with patch.object(archive.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.session.save_step(1, b"data", {})
        self.assertFalse((self.session.directory / "Step001.webp").exists())
        self.assertFalse(list(self.session.directory.glob("*.part")))

    def test_manifest_failure_preserves_saved_file_for_finalization(self):
        with patch.object(self.session, "persist", side_effect=OSError("manifest unavailable")):
            with self.assertRaises(OSError):
                self.session.save_step(1, b"saved", {})
        result = self.session.finalize("interrupted")
        self.assertEqual([a["step"] for a in result["artifacts"]], [1])
        with ZipFile(self.session.directory / "MyShot.zip") as bundle:
            self.assertEqual(bundle.read("MyShot/Step001.webp"), b"saved")

    def test_wav_write_failure_preserves_webp_for_finalization(self):
        original = archive._atomic_write

        def write(path, data):
            if path.suffix == ".wav":
                raise OSError("disk full")
            return original(path, data)

        with patch.object(archive, "_atomic_write", side_effect=write):
            with self.assertRaises(OSError):
                self.session.save_step(6, b"saved", {}, b"audio", {})
        self.session.finalize("interrupted")
        with ZipFile(self.session.directory / "MyShot.zip") as bundle:
            self.assertEqual(bundle.namelist(), ["MyShot/Step006.webp"])
            self.assertEqual(bundle.read("MyShot/Step006.webp"), b"saved")

    def test_zip_failure_keeps_individual_files(self):
        self.session.save_step(1, b"saved", {})
        with patch.object(archive.ZipFile, "write", side_effect=OSError("zip failed")):
            result = self.session.finalize("completed")
        self.assertEqual(result["archive_state"], "unavailable")
        self.assertEqual((self.session.directory / "Step001.webp").read_bytes(), b"saved")
        self.assertFalse((self.session.directory / "MyShot.zip").exists())

    def test_restart_can_read_finalized_archive_and_detect_abandoned_run(self):
        self.session.save_step(1, b"saved", {})
        with patch.object(archive, "_PROCESS_ID", "new process"):
            result = archive.read_manifest(self.temp.name, "MyShot", self.session.manifest["run_id"])
        self.assertEqual(result["archive_state"], "unavailable")
        self.session.finalize("completed")
        with patch.object(archive, "_PROCESS_ID", "new process"):
            result = archive.read_manifest(self.temp.name, "MyShot", self.session.manifest["run_id"])
            path = archive.artifact_path(self.temp.name, "MyShot", result["run_id"], result)
        self.assertTrue(path.is_file())
        self.assertNotIn("process_id", result)


def load_preview_modules():
    """Import actual code with only the host/GPU dependencies replaced."""
    modules = {}

    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__dict__.update(attributes)
        modules[name] = result
        if "." in name and name.rsplit(".", 1)[0] in modules:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, result)
        return result

    package_name = "preview_tests_package"
    module(package_name, __path__=[str(HERE)])
    modules[package_name + ".minimax_preview_archive"] = archive
    if preview_audio:
        modules[package_name + ".minimax_preview_audio"] = preview_audio
    module("torch", Tensor=object)
    module("torch.nn")
    module("torch.nn.functional")
    module("numpy")
    module("av")
    module("comfy")
    module("comfy.patcher_extension")
    module("comfy.utils")

    class Interrupted(BaseException):
        pass

    module("comfy.model_management", InterruptProcessingException=Interrupted)
    module("comfy_execution")
    module("comfy_execution.utils", get_executing_context=lambda: types.SimpleNamespace(
        prompt_id="prompt-a", node_id="sampler-a"))
    module("folder_paths", get_output_directory=lambda: "unused", get_temp_directory=lambda: "unused")

    class LatentPreviewer:
        def decode_latent_to_preview_image(self, *args):
            return "original"

    module("latent_preview", LatentPreviewer=LatentPreviewer)
    events = []
    routes = web.RouteTableDef() if web else types.SimpleNamespace(
        get=lambda *args: lambda f: f, post=lambda *args: lambda f: f)
    srv = types.SimpleNamespace(client_id="client-a", routes=routes,
                                send_sync=lambda event, data, sid: events.append((event, data, sid)))
    module("server", PromptServer=types.SimpleNamespace(instance=srv))
    module("protocol", BinaryEventTypes=types.SimpleNamespace(PREVIEW_IMAGE=1))
    module("comfy_api")
    input_type = types.SimpleNamespace(
        Input=lambda name, **kw: types.SimpleNamespace(id=name, **kw),
        Output=lambda **kw: types.SimpleNamespace(**kw))
    schema_io = types.SimpleNamespace(ComfyNode=object, NodeOutput=tuple,
        Schema=lambda **kw: types.SimpleNamespace(**kw),
        Hidden=types.SimpleNamespace(unique_id="unique_id"),
        **{name: input_type for name in ("Model", "Vae", "Combo", "Int", "Float", "Boolean", "String")})
    module("comfy_api.latest", io=schema_io)
    loaded = []
    with patch.dict(sys.modules, modules):
        for name in (["minimax_preview", "minimax_media"] if web else ["minimax_preview"]):
            spec = importlib.util.spec_from_file_location(package_name + "." + name, HERE / (name + ".py"))
            obj = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = obj
            spec.loader.exec_module(obj)
            loaded.append(obj)
    # Lazy codec imports happen during callbacks/HTTP requests, after host stubs exit.
    sys.modules[package_name] = modules[package_name]
    if preview_audio:
        sys.modules[package_name + ".minimax_preview_audio"] = preview_audio
    return loaded, srv, events


@unittest.skipUnless(Image, "Pillow is needed for WebP and sampler lifecycle tests")
class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules, cls.server, cls.events = load_preview_modules()
        cls.preview = modules[0]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events.clear()
        self.server.client_id = "client-a"
        self.preview.folder_paths.get_output_directory = lambda: self.temp.name
        self.frames = [Image.new("RGB", (12, 8), c) for c in ("red", "blue")]
        self.patchers = [
            patch.object(self.preview, "_RGBFactors", return_value=lambda video: video),
            patch.object(self.preview, "_pick_frames", side_effect=lambda v, n: v),
            patch.object(self.preview, "_to_pil", return_value=self.frames),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def sample(self, interval=1, stop=None, error=None, target="node", wrapper=None):
        preview = self.preview
        self.callbacks = []
        class Video(types.SimpleNamespace):
            def __getitem__(self, key):
                return self
        video = Video(ndim=5, shape=(1, 4, 2, 8, 12))

        class Executor:
            class_obj = types.SimpleNamespace(model_patcher=types.SimpleNamespace(
                model=types.SimpleNamespace(latent_format=None)))

            def __call__(executor_self, noise, latent, sampler, sigmas, mask, callback, *args, **kwargs):
                for step in range(30):
                    if step == stop:
                        raise error
                    callback(step, video, video, 30)
                return "sampled"

        wrapper = wrapper or preview._OuterSampleWrapper(
            "7", preview.DECODE_FAST, None, 512, 24, 24, 80, interval, True, target,
            max_overhead=1, archive_folder_name="MyShot")
        return wrapper(Executor(), None, None, None, None, None,
                       lambda *args: self.callbacks.append(args[0]), False, 42)

    def manifests(self):
        return [json.loads(p.read_text()) for p in Path(self.temp.name).rglob("manifest.json")]

    def test_every_step_is_saved_despite_display_throttle(self):
        self.assertEqual(self.sample(), "sampled")
        manifest, = self.manifests()
        self.assertEqual([a["step"] for a in manifest["artifacts"]], list(range(1, 31)))
        self.assertEqual(self.callbacks, list(range(30)))
        displays = [e for e in self.events if e[0] == self.preview.EVENT]
        self.assertLess(len(displays), 30)
        self.assertTrue(all(sid == "client-a" for _, _, sid in self.events))
        path = next(Path(self.temp.name).rglob("Step001.webp"))
        with Image.open(path) as img:
            self.assertEqual(img.n_frames, 2)
            img.seek(1)
            self.assertGreater(img.convert("RGB").getpixel((0, 0))[2], 200)

    def test_interval_and_final_callback(self):
        self.sample(interval=5)
        self.assertEqual([a["step"] for a in self.manifests()[0]["artifacts"]], [1, 6, 11, 16, 21, 26, 30])

    def test_cancel_preserves_zip_and_original_exception(self):
        exc = self.preview.InterruptProcessingException()
        original = self.preview.latent_preview.LatentPreviewer.decode_latent_to_preview_image
        with self.assertRaises(type(exc)) as caught:
            self.sample(stop=3, error=exc)
        self.assertIs(caught.exception, exc)
        self.assertIs(self.preview.latent_preview.LatentPreviewer.decode_latent_to_preview_image, original)
        manifest, = self.manifests()
        self.assertEqual(manifest["sampling_status"], "interrupted")
        self.assertEqual(len(manifest["artifacts"]), 3)
        with ZipFile(next(Path(self.temp.name).rglob("MyShot.zip"))) as bundle:
            self.assertEqual(len(bundle.namelist()), 3)

    def test_cancel_during_decode(self):
        exc = self.preview.InterruptProcessingException()
        wrapper = self.preview._OuterSampleWrapper("7", self.preview.DECODE_VAE, object(),
            512, 24, 24, 80, 1, True, "node", archive_folder_name="MyShot")
        with patch.object(self.preview, "_vae_decode", side_effect=exc):
            with self.assertRaises(type(exc)):
                self.sample(wrapper=wrapper)
        self.assertEqual(self.manifests()[0]["sampling_status"], "interrupted")
        self.assertEqual(self.callbacks, [])
        with ZipFile(next(Path(self.temp.name).rglob("MyShot.zip"))) as bundle:
            self.assertEqual(bundle.namelist(), ["MyShot/"])

    def test_sampler_error_and_zip_error_do_not_mask_original(self):
        exc = RuntimeError("sampler failed")
        with patch.object(archive.ZipFile, "write", side_effect=OSError("zip failed")):
            with self.assertRaises(RuntimeError) as caught:
                self.sample(stop=2, error=exc)
        self.assertIs(caught.exception, exc)
        manifest, = self.manifests()
        self.assertEqual(manifest["sampling_status"], "failed")
        self.assertEqual(manifest["archive_state"], "unavailable")

    def test_cached_wrapper_creates_distinct_sessions(self):
        wrapper = self.preview._OuterSampleWrapper("7", self.preview.DECODE_FAST, None,
            512, 24, 24, 80, 5, True, "node", archive_folder_name="MyShot")
        self.sample(wrapper=wrapper)
        self.sample(wrapper=wrapper)
        self.assertEqual(len(self.manifests()), 2)

    def test_vhs_and_headless_also_archive(self):
        for target in ("sampler (VHS)", "both"):
            with self.subTest(target=target):
                self.sample(interval=5, target=target)
        self.server.client_id = None
        self.events.clear()
        self.sample(interval=5)
        self.assertEqual(self.events, [])
        self.assertEqual(len(self.manifests()), 3)

    def test_encoder_failure_records_missing_capture(self):
        with patch.object(self.preview, "_encode_animated_webp", side_effect=[None] + [b"encoded"] * 29):
            self.sample()
        manifest, = self.manifests()
        self.assertEqual(manifest["capture_failures"][0]["step"], 1)
        self.assertEqual(manifest["artifacts"][0]["step"], 2)
        self.assertEqual(self.callbacks, list(range(30)))

    def test_storage_failure_stops_writes_but_sampling_continues(self):
        with patch.object(archive.PreviewArchiveSession, "save_step", side_effect=OSError("disk full")) as save:
            self.sample()
        self.assertEqual(save.call_count, 1)
        self.assertEqual(len(self.manifests()[0]["capture_failures"]), 30)
        self.assertEqual(self.callbacks, list(range(30)))

    def test_unwritable_output_keeps_live_preview_and_sampling(self):
        with patch.object(self.preview, "PreviewArchiveSession", side_effect=OSError("read only")):
            self.assertEqual(self.sample(), "sampled")
        self.assertEqual(self.manifests(), [])
        self.assertTrue(any(data.get("webp") for event, data, sid in self.events if event == self.preview.EVENT))
        self.assertEqual(self.events[-1][1]["archive_state"], "unavailable")

    def test_notification_failure_does_not_stop_disk_archiving(self):
        with patch.object(self.server, "send_sync", side_effect=OSError("disconnected")):
            with self.assertLogs(self.preview.log, level="WARNING"):
                self.assertEqual(self.sample(interval=5), "sampled")
        self.assertEqual(self.manifests()[0]["archive_state"], "ready")

    def test_manifest_finalization_failure_does_not_mask_cancellation(self):
        exc = self.preview.InterruptProcessingException()
        original_finalize = archive.PreviewArchiveSession.finalize

        def fail_manifest(session, *args):
            with patch.object(session, "persist", side_effect=OSError("manifest failed")):
                return original_finalize(session, *args)

        with patch.object(archive.PreviewArchiveSession, "finalize", fail_manifest):
            with self.assertLogs(self.preview.log, level="WARNING"):
                with self.assertRaises(type(exc)) as caught:
                    self.sample(stop=2, error=exc)
        self.assertIs(caught.exception, exc)
        self.assertEqual(self.events[-1][1]["archive_state"], "unavailable")
        self.assertTrue(next(Path(self.temp.name).rglob("MyShot.zip")).is_file())

    def test_folder_widget_is_last_and_defaulted_in_execute(self):
        import inspect
        node = self.preview.MiniMaxH3PreviewOverride
        schema = node.define_schema()
        self.assertEqual(schema.inputs[-1].id, "audio_vae")
        self.assertEqual(schema.inputs[-2].id, "archive_folder_name")
        self.assertEqual(schema.inputs[-3].id, "playback")
        self.assertEqual(inspect.signature(node.execute).parameters["archive_folder_name"].default,
                         archive.DEFAULT_FOLDER)

    def test_audio_capture_uses_same_step_and_survives_cancellation(self):
        wrapper = self.preview._OuterSampleWrapper("7", self.preview.DECODE_FAST, None,
            512, 24, 24, 80, 1, True, "node", archive_folder_name="MyShot", audio_vae=object())
        with patch.object(self.preview, "_decode_audio_wav", return_value=(b"wav", {"duration": 5 / 24})):
            with self.assertRaises(self.preview.InterruptProcessingException):
                self.sample(wrapper=wrapper, stop=3, error=self.preview.InterruptProcessingException())
        manifest, = self.manifests()
        self.assertEqual([a["audio"]["filename"] for a in manifest["artifacts"]],
                         ["Step001.wav", "Step002.wav", "Step003.wav"])
        with ZipFile(next(Path(self.temp.name).rglob("MyShot.zip"))) as bundle:
            self.assertEqual(len(bundle.namelist()), 6)

    def test_audio_failure_keeps_video_steps(self):
        wrapper = self.preview._OuterSampleWrapper("7", self.preview.DECODE_FAST, None,
            512, 24, 24, 80, 5, True, "node", archive_folder_name="MyShot", audio_vae=object())
        with patch.object(self.preview, "_decode_audio_wav", side_effect=ValueError("bad audio")):
            with self.assertLogs(self.preview.log, level="WARNING"):
                self.sample(wrapper=wrapper)
        manifest, = self.manifests()
        self.assertEqual(len(manifest["artifacts"]), 7)
        self.assertEqual(len(manifest["capture_failures"]), 7)
        self.assertTrue(all(a.get("audio_error") and not a.get("audio") for a in manifest["artifacts"]))

    def test_cancel_during_audio_is_not_swallowed(self):
        wrapper = self.preview._OuterSampleWrapper("7", self.preview.DECODE_FAST, None,
            512, 24, 24, 80, 1, True, "node", archive_folder_name="MyShot", audio_vae=object())
        exc = self.preview.InterruptProcessingException()
        with patch.object(self.preview, "_decode_audio_wav", side_effect=exc):
            with self.assertRaises(type(exc)) as caught:
                self.sample(wrapper=wrapper)
        self.assertIs(caught.exception, exc)
        self.assertEqual(self.manifests()[0]["sampling_status"], "interrupted")

    def test_audio_stream_unpacking(self):
        video, audio = object(), object()
        nested = types.SimpleNamespace(is_nested=True, tensors=(video, audio))
        self.assertIs(self.preview._audio_stream(nested), audio)
        with patch.object(self.preview.comfy.utils, "unpack_latents", return_value=(video, audio), create=True):
            self.assertIs(self.preview._audio_stream(object(), [(1, 24, 2, 8, 8), (1, 32, 2, 8)]), audio)


@unittest.skipUnless(web and Image, "aiohttp and Pillow are needed for HTTP tests")
class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        modules, srv, _ = load_preview_modules()
        self.media = modules[1]
        self.media.folder_paths.get_output_directory = lambda: self.temp.name
        self.media.folder_paths.get_temp_directory = lambda: str(Path(self.temp.name) / "temp")
        app = web.Application()
        app.add_routes(srv.routes)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.session = archive.PreviewArchiveSession(self.temp.name, "été 東京", "7")
        data = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(data, format="WEBP")
        self.data = data.getvalue()
        self.session.save_step(1, self.data, {})

    def url(self, kind, **extra):
        from urllib.parse import urlencode
        return "/minimax_director/preview_archive/" + kind + "?" + urlencode({
            "folder": self.session.manifest["folder_name"],
            "run_id": self.session.manifest["run_id"], **extra})

    async def test_downloads_match_disk_and_zip(self):
        response = await self.client.get(self.url("step", step=1, download=1))
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), self.data)
        self.assertEqual(response.content_type, "image/webp")
        self.assertIn("attachment", response.headers["Content-Disposition"])
        response = await self.client.get(self.url("zip"))
        self.assertEqual(response.status, 409)
        self.session.finalize("interrupted")
        response = await self.client.get(self.url("zip"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "application/zip")
        with ZipFile(io.BytesIO(await response.read())) as bundle:
            self.assertEqual(bundle.read("été 東京/Step001.webp"), self.data)
        response = await self.client.get(self.url("manifest"))
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual((await response.json())["sampling_status"], "interrupted")

    async def test_bad_paths_and_missing_steps(self):
        for parameters, expected in (({"step": "../secret"}, 400), ({"step": 2}, 404),
                                     ({"step": 0}, 400), ({"folder": "../outside", "step": 1}, 400),
                                     ({"run_id": "manifest.json.part", "step": 1}, 400)):
            response = await self.client.get(self.url("step", **parameters))
            self.assertEqual(response.status, expected)

    @unittest.skipUnless(preview_audio, "PyAV and NumPy needed for paired playback tests")
    async def test_audio_and_synchronized_playback_routes(self):
        samples = preview_audio.np.zeros((2, 32000), dtype="float32")
        wav, meta = preview_audio.encode_wav(samples, 32000, 1)
        webp = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(webp, format="WEBP", save_all=True,
            append_images=[Image.new("RGB", (16, 16), "blue")], duration=100, loop=0)
        self.session.save_step(2, webp.getvalue(), {}, wav, meta)
        response = await self.client.get(self.url("audio", step=2, download=1))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "audio/wav")
        self.assertEqual(await response.read(), wav)
        response = await self.client.get(self.url("audio", step=1))
        self.assertEqual(response.status, 404)
        response = await self.client.get(self.url("playback", step=2))
        self.assertEqual(response.status, 200, await response.text() if response.status != 200 else "")
        self.assertEqual(response.content_type, "video/mp4")
        playback = await response.read()
        with preview_audio.av.open(io.BytesIO(playback)) as container:
            self.assertEqual(len(container.streams.audio), 1)
            self.assertEqual(len(container.streams.video), 1)
        response = await self.client.get(self.url("playback", step=2), headers={"Range": "bytes=0-31"})
        self.assertEqual(response.status, 206)
        self.assertEqual(await response.read(), playback[:32])
        self.session.finalize("interrupted")
        response = await self.client.get(self.url("zip"))
        with ZipFile(io.BytesIO(await response.read())) as bundle:
            self.assertIn("été 東京/Step002.wav", bundle.namelist())
            self.assertFalse(any(n.endswith(".mp4") for n in bundle.namelist()))


@unittest.skipUnless(preview_audio, "PyAV and NumPy needed for audio tests")
class AudioTests(unittest.TestCase):
    def test_wav_channels_timing_and_input_unchanged(self):
        import wave
        np = preview_audio.np
        samples = np.array([[2.0, 0.5, -2.0], [-0.25, 0.0, 0.25]], dtype=np.float32)
        original = samples.copy()
        data, metadata = preview_audio.encode_wav(samples, 32000, 0.01)
        np.testing.assert_array_equal(samples, original)
        with wave.open(io.BytesIO(data)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getnframes()), (2, 32000, 320))
            self.assertEqual(wav.getsampwidth(), 2)
            pcm = np.frombuffer(wav.readframes(320), dtype="<i2").reshape(-1, 2)
        self.assertEqual(tuple(pcm[0]), (32767, -8192))
        self.assertTrue((pcm[3:] == 0).all())
        self.assertEqual(metadata["duration"], .01)
        with self.assertRaises(ValueError):
            preview_audio.encode_wav(np.full((2, 10), np.nan), 32000, 1)

    def test_pair_zip_and_real_time_mux_preserve_merged_frame_timing(self):
        np = preview_audio.np
        with tempfile.TemporaryDirectory() as output:
            session = archive.PreviewArchiveSession(output, "AudioShot", "7")
            t = np.arange(64000) / 32000
            samples = np.stack([.2 * np.sin(t * 440 * 2 * np.pi), .1 * np.sin(t * 220 * 2 * np.pi)])
            wav, metadata = preview_audio.encode_wav(samples, 32000, 2)
            webp = io.BytesIO()
            Image.new("RGB", (32, 24), "red").save(webp, format="WEBP", save_all=True,
                append_images=[Image.new("RGB", (32, 24), "red"), Image.new("RGB", (32, 24), "blue")],
                duration=[100, 100, 200], loop=0)
            session.save_step(6, webp.getvalue(), {}, wav, metadata)
            destination = Path(output) / "temp" / "preview.mp4"
            preview_audio.build_av_preview(session.directory / "Step006.webp",
                                            session.directory / "Step006.wav", destination)
            with preview_audio.av.open(str(destination)) as container:
                audio, video = container.streams.audio[0], container.streams.video[0]
                self.assertAlmostEqual(float(audio.duration * audio.time_base), 2, delta=.05)
                self.assertAlmostEqual(float(video.duration * video.time_base), 2, delta=.05)
                frames = list(container.decode(video))
                self.assertAlmostEqual(float(frames[0].pts * frames[0].time_base), 0, delta=.01)
                self.assertAlmostEqual(float(frames[-1].pts * frames[-1].time_base), 1, delta=.01)
                self.assertGreater(frames[0].to_image().getpixel((0, 0))[0], 200)
                self.assertGreater(frames[-1].to_image().getpixel((0, 0))[2], 200)
            session.finalize("interrupted")
            with ZipFile(session.directory / "AudioShot.zip") as bundle:
                self.assertEqual(bundle.namelist(), ["AudioShot/Step006.webp", "AudioShot/Step006.wav"])
                self.assertEqual(bundle.read("AudioShot/Step006.wav"), wav)

    def test_audio_decoder_receives_a_private_copy(self):
        np = preview_audio.np
        modules, _, _ = load_preview_modules()
        preview = modules[0]

        class Tensor:
            def __init__(self, data):
                self.data = np.asarray(data)
                self.ndim = self.data.ndim

            def __getitem__(self, key): return Tensor(self.data[key])
            def detach(self): return self
            def clone(self): return Tensor(self.data.copy())
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.data

        latent = Tensor(np.ones((1, 32, 2, 40), dtype=np.float32))

        def decode(vae, samples):
            samples["samples"].data[:] = 99
            return types.SimpleNamespace(args=[{"waveform": Tensor(np.zeros((1, 2, 32000))), "sample_rate": 32000}])

        nodes = types.SimpleNamespace(VAEDecodeAudio=types.SimpleNamespace(execute=decode))
        with patch.object(preview, "audio_nodes", return_value=nodes):
            data, meta = preview._decode_audio_wav(object(), latent, 1)
        self.assertTrue((latent.data == 1).all())
        self.assertEqual(meta["duration"], 1)
        self.assertTrue(data.startswith(b"RIFF"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
