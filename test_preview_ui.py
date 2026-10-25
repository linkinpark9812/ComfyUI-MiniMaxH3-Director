"""Real-browser preview widget checks using a simulated ComfyUI host.

Requires Pillow, aiohttp, Playwright, and Chrome (or set MMX_TEST_BROWSER to a
Playwright browser channel). Run: python test_preview_ui.py
This exercises the actual extension and HTTP routes, not real H3 GPU sampling.
"""

import io
import os
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image
from playwright.async_api import async_playwright

import minimax_preview_archive as archive
from test_preview_archive import HERE, load_preview_modules, preview_audio

HOST = """<!doctype html><html><head><meta charset="utf-8"></head>
<body style="background:#222;font-family:Arial;color:#ddd">
<h3>MiniMax H3 Preview Override</h3>
<div id="node" style="width:560px;height:330px;resize:both;overflow:hidden;border:1px solid #666;padding:8px"></div>
<script>
const api = new EventTarget();
api.apiURL = path => '/prefix' + path;
api.fetchApi = (path, options) => fetch(api.apiURL(path), options);
const app = { graph: { _nodes: [] }, registerExtension: ext => window.extension = ext };
window.comfyAPI = { app: { app }, api: { api } };
window.emit = (type, data) => api.dispatchEvent(new CustomEvent(type, { detail: data }));
window.ready = async () => {
  function Node() {
    this.id = 7; this.size = [340, 520]; this.widgets = [];
    this.addDOMWidget = (name, type, root) => { document.querySelector('#node').append(root); return {}; };
    this.computeSize = () => [340, 520]; this.setDirtyCanvas = () => {};
    this.onNodeCreated();
  }
  await extension.beforeRegisterNodeDef(Node, { name: 'MiniMaxH3PreviewOverrideCS' });
  window.node = new Node(); app.graph._nodes.push(node);
  await extension.setup();
};
</script><script type="module">import '/prefix/extension.js'; await window.ready();</script></body></html>"""


class PreviewBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        modules, srv, _ = load_preview_modules()
        modules[1].folder_paths.get_output_directory = lambda: self.temp.name
        modules[1].folder_paths.get_temp_directory = lambda: str(Path(self.temp.name) / "cache")
        app = web.Application()
        app.add_routes(srv.routes)

        async def script(request):
            return web.Response(text=(HERE / "js/minimax_preview.js").read_text(encoding="utf-8"),
                                content_type="text/javascript")

        async def index(request):
            return web.Response(text=HOST, content_type="text/html")

        app.router.add_get("/extension.js", script)
        parent = web.Application()
        parent.add_subapp("/prefix", app)
        parent.router.add_get("/", index)
        self.server = TestServer(parent)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.playwright = await async_playwright().start()
        self.addAsyncCleanup(self.playwright.stop)
        self.browser = await self.playwright.chromium.launch(
            channel=os.environ.get("MMX_TEST_BROWSER", "chrome"), headless=True, timeout=15000)
        self.addAsyncCleanup(self.browser.close)
        self.page = await self.browser.new_page(accept_downloads=True)
        self.page.set_default_timeout(10000)
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        await self.page.goto(str(self.server.make_url("/")))
        try:
            await self.page.wait_for_function("window.node?._mmxPreview")
        except Exception as exc:
            raise AssertionError(f"Widget failed to initialize; browser errors: {self.errors}") from exc
        self.sequence = 0
        self.session = self.new_session()
        data = io.BytesIO()
        Image.new("RGB", (48, 32), "red").save(data, format="WEBP", save_all=True,
            append_images=[Image.new("RGB", (48, 32), "blue")], duration=200, loop=0)
        self.webp = data.getvalue()

    def new_session(self, prompt="prompt-a"):
        return archive.PreviewArchiveSession(self.temp.name, "My Shot 東京", "7",
            {"prompt_id": prompt}, {"preview_target": "node"})

    async def emit(self, kind, payload=None, event="minimax_h3_preview_archive", session=None):
        self.sequence += 1
        session = session or self.session
        data = {key: session.manifest.get(key) for key in ("node_id", "run_id", "folder_name", "prompt_id")}
        data.update(payload or {})
        data.update(kind=kind, sequence=self.sequence)
        await self.page.evaluate("([type, data]) => emit(type, data)", [event, data])

    async def start(self):
        await self.page.evaluate("emit('execution_start', {prompt_id: 'prompt-a'})")
        await self.emit("started", self.session.snapshot())

    async def save(self, step, notify=True):
        record = self.session.save_step(step, self.webp, {"frames": 2, "fps": 5,
            "source_fps": 24, "total_steps": 30, "ms": 10, "mode": "latent2rgb (fast)"})
        if notify:
            await self.emit("step_saved", {"artifact": record})
        return record

    async def test_history_selection_downloads_cancel_and_new_run(self):
        await self.start()
        first = await self.save(1)
        await self.emit("display", {"artifact": first}, event="minimax_h3_preview")
        source = await self.page.locator("#node img").get_attribute("src")
        second = await self.save(6)
        self.assertEqual(await self.page.locator("#node img").get_attribute("src"), source)
        async with self.page.expect_download() as download:
            await self.page.get_by_role("link", name="Download Step006.webp", exact=True).click()
        artifact = await download.value
        self.assertEqual(artifact.suggested_filename, "Step006.webp")
        self.assertEqual(Path(await artifact.path()).read_bytes(), self.webp)
        await self.emit("display", {"artifact": second}, event="minimax_h3_preview")
        await self.page.get_by_role("button", name="Step001.webp", exact=True).click()
        self.assertIn("step=1", await self.page.locator("#node img").get_attribute("src"))
        await self.page.get_by_role("button", name="Live", exact=True).click()
        self.assertIn("step=6", await self.page.locator("#node img").get_attribute("src"))
        await self.emit("finished", self.session.finalize("interrupted"))
        self.assertIn("Canceled", await self.page.locator("[aria-live]").inner_text())
        async with self.page.expect_download() as download:
            await self.page.get_by_role("button", name="Download all (.zip)", exact=True).click()
        bundle = await download.value
        self.assertEqual(bundle.suggested_filename, "My Shot 東京.zip")
        with ZipFile(await bundle.path()) as zipped:
            self.assertEqual(zipped.namelist(), ["My Shot 東京/Step001.webp", "My Shot 東京/Step006.webp"])
        screenshot = os.environ.get("MMX_TEST_SCREENSHOT")
        if screenshot:
            await self.page.screenshot(path=str(Path(screenshot).with_stem("preview-history")))
        await self.page.evaluate("emit('execution_start', {prompt_id: 'prompt-b'})")
        self.assertEqual(await self.page.locator("[data-step]").count(), 0)
        self.assertIsNone(await self.page.locator("#node img").get_attribute("src"))
        await self.emit("finished", self.session.snapshot())  # Late old execution.
        self.assertEqual(await self.page.locator("[data-step]").count(), 0)
        self.assertTrue((self.session.directory / "My Shot 東京.zip").exists())
        self.assertEqual(self.errors, [])

    async def test_reconnect_recovers_missing_steps_and_finished_archive(self):
        await self.start()
        await self.save(1)
        await self.save(6, notify=False)
        self.session.finalize("completed")
        await self.page.evaluate("emit('reconnected', {})")
        await self.page.wait_for_function("node._mmxPreview.artifacts.size === 2 && node._mmxPreview.terminal")
        self.assertEqual(await self.page.locator('[role="button"][aria-disabled="false"]').count(), 1)
        self.assertEqual(self.errors, [])

    async def test_multiple_sampler_invocations_and_stale_events(self):
        await self.start()
        await self.save(1)
        previous = self.session
        self.session = self.new_session()
        await self.emit("started", self.session.snapshot())
        await self.save(6)
        await self.emit("finished", previous.finalize("completed"), session=previous)
        self.assertEqual(await self.page.locator("[data-step]").count(), 1)
        self.assertEqual(await self.page.locator("[data-step]").get_attribute("data-step"), "6")
        self.assertEqual(self.errors, [])

    async def test_long_history_resizes_without_growing_node(self):
        await self.start()
        records = [{"step": n, "filename": f"Step{n:03d}.webp"} for n in range(1, 1001)]
        await self.emit("started", {**self.session.snapshot(), "artifacts": records})
        bounds = await self.page.locator("#node").bounding_box()
        self.assertLess(bounds["height"], 355)
        self.assertEqual(await self.page.locator("[data-step]").count(), 1000)
        await self.page.evaluate("document.querySelector('#node').style.width='340px'")
        await self.page.evaluate("document.querySelector('#node').style.height='180px'")
        self.assertTrue(await self.page.evaluate("""() => {
          const p = node._mmxPreview;
          return p.list.scrollHeight > p.list.clientHeight && p.root.scrollHeight <= p.root.clientHeight;
        }"""))
        screenshot = os.environ.get("MMX_TEST_SCREENSHOT")
        if screenshot:
            await self.page.screenshot(path=screenshot)
        self.assertEqual(self.errors, [])

    async def test_cached_execution_and_removed_node(self):
        await self.page.evaluate("emit('execution_start', {prompt_id: 'prompt-a'})")
        await self.page.evaluate("emit('execution_success', {prompt_id: 'prompt-a'})")
        self.assertIn("No new previews generated", await self.page.locator("[aria-live]").inner_text())
        await self.start()
        await self.save(1)
        await self.page.evaluate("node.onRemoved()")
        await self.emit("finished", self.session.finalize("completed"))
        self.assertTrue(await self.page.evaluate("node._mmxPreview === null"))
        self.assertEqual(self.errors, [])

    @unittest.skipUnless(preview_audio, "PyAV and NumPy needed for AV playback")
    async def test_paired_audio_playback_mute_download_and_reset(self):
        await self.start()
        samples = preview_audio.np.zeros((2, 64000), dtype="float32")
        wav, meta = preview_audio.encode_wav(samples, 32000, 2)
        record = self.session.save_step(6, self.webp, {"frames": 2, "fps": 5,
            "total_steps": 30, "source_fps": 24}, wav, meta)
        await self.emit("step_saved", {"artifact": record})
        await self.emit("display", {"artifact": record}, event="minimax_h3_preview")
        await self.page.wait_for_function("node._mmxPreview.video.readyState >= 2")
        self.assertTrue(await self.page.evaluate("node._mmxPreview.video.muted"))
        duration = await self.page.evaluate("node._mmxPreview.video.duration")
        self.assertAlmostEqual(duration, 2, delta=.06)
        await self.page.get_by_role("button", name="Unmute", exact=True).click()
        self.assertFalse(await self.page.evaluate("node._mmxPreview.video.muted"))
        await self.page.get_by_role("button", name="Mute", exact=True).click()
        self.assertTrue(await self.page.evaluate("node._mmxPreview.video.muted"))
        async with self.page.expect_download() as download:
            await self.page.get_by_role("link", name="Download Step006.wav", exact=True).click()
        file = await download.value
        self.assertEqual(file.suggested_filename, "Step006.wav")
        self.assertEqual(Path(await file.path()).read_bytes(), wav)
        await self.emit("finished", self.session.finalize("interrupted"))
        async with self.page.expect_download() as download:
            await self.page.get_by_role("button", name="Download all (.zip)", exact=True).click()
        file = await download.value
        with ZipFile(await file.path()) as bundle:
            self.assertEqual(bundle.namelist(), ["My Shot 東京/Step006.webp", "My Shot 東京/Step006.wav"])
        screenshot = os.environ.get("MMX_TEST_SCREENSHOT")
        if screenshot:
            await self.page.screenshot(path=str(Path(screenshot).with_stem("preview-audio")))
        await self.page.evaluate("emit('execution_start', {prompt_id: 'next'})")
        self.assertTrue(await self.page.evaluate("node._mmxPreview.video.paused"))
        self.assertIsNone(await self.page.locator("#node video").get_attribute("src"))
        self.assertEqual(self.errors, [])

    @unittest.skipUnless(preview_audio, "PyAV and NumPy needed for AV playback")
    async def test_audio_step_switch_and_playback_failure_fallback(self):
        await self.start()
        wav, meta = preview_audio.encode_wav(preview_audio.np.zeros((2, 32000)), 32000, 1)
        for step in (1, 6):
            record = self.session.save_step(step, self.webp, {"frames": 2, "fps": 5,
                "total_steps": 30}, wav, meta)
            await self.emit("step_saved", {"artifact": record})
        await self.page.get_by_role("button", name="Step001.webp", exact=True).click()
        await self.page.wait_for_function("node._mmxPreview.video.readyState >= 2")
        await self.page.get_by_role("button", name="Unmute", exact=True).click()
        await self.page.get_by_role("button", name="Step006.webp", exact=True).click()
        await self.page.wait_for_function("node._mmxPreview.video.readyState >= 2")
        self.assertIn("step=6", await self.page.locator("#node video").get_attribute("src"))
        self.assertFalse(await self.page.evaluate("node._mmxPreview.video.muted"))
        # A failed playback copy still leaves both original downloads usable.
        await self.page.route("**/preview_archive/playback?**step=1", lambda route: route.abort())
        await self.page.get_by_role("button", name="Step001.webp", exact=True).click()
        await self.page.wait_for_function("node._mmxPreview.sound.disabled")
        self.assertTrue(await self.page.locator("#node img").is_visible())
        self.assertFalse(await self.page.locator("#node video").is_visible())
        self.assertIn("AV playback unavailable", await self.page.locator("[aria-live]").inner_text())
        self.assertEqual(await self.page.get_by_role("link", name="Download Step001.wav", exact=True).count(), 1)
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
