# MiniMax H3 preview step history and downloads

Status: implemented October 25, 2026; offline and browser-fixture validation completed.
Real ComfyUI/H3 GPU acceptance remains environment-dependent. See the implementation record below.

## Goal and feasibility

Enhance **MiniMax H3 Preview Override** (`MiniMaxH3PreviewOverrideCS`) with a right-hand history of downloadable animated WebP previews, a downloadable ZIP after sampling completes or is canceled, and permanent server-side copies of both the individual files and ZIP.

This is a contained, moderately sized change. The existing node already decodes sampling callbacks into animated WebP previews. The UI list and ZIP creation are straightforward; the substantial work is separating capture from playback throttling, finalizing reliably on cancellation, and isolating repeated runs. No new third-party dependency should be necessary: use Python's filesystem and `zipfile` facilities, the existing Pillow encoder, and existing aiohttp/ComfyUI interfaces.

The original deliverable was this plan. The user subsequently authorized implementation; the design and acceptance criteria remain here as the implementation reference.

## Confirmed requirements and behavior decisions

1. Show the current animated preview on the left and a scrollable step/download list on the right.
2. Allow individual WebP downloads as soon as each file has been saved.
3. Provide **Download all (.zip)** once the sampling invocation finishes or is canceled and the ZIP has actually been finalized.
4. Add a node setting for the folder name used inside the ZIP and in server storage.
5. Use `.webp`, interpreting the initial `Step001.web` example as a filename-extension typo.
6. Name files using the actual one-based sampler step: `Step001.webp`, `Step002.webp`, ..., `Step1000.webp`. Formatting is `f"Step{step:03d}.webp"`: three digits minimum, without truncating larger numbers.
7. The user explicitly confirmed capture defaults to **every 1 sampling step**, with a configurable **every X steps** interval. Skipped steps remain visible as gaps in filenames; never renumber saved previews consecutively.
8. Reuse the existing `every_n_steps` input, whose default is already 1. Do not add a competing interval setting.
9. Retain the current interval anchor: capture step 1, then `1 + N`, `1 + 2N`, and so on. This preserves the current `step % every_n_steps == 0` callback rule, where callbacks are zero-based. Also capture the final callback on successful sampling even when off cadence. For 30 steps with interval 5, filenames are `Step001.webp`, `Step006.webp`, `Step011.webp`, `Step016.webp`, `Step021.webp`, `Step026.webp`, `Step030.webp`. Explain this rule in the tooltip and README.
10. Clear the node's visible history for a new execution, while retaining every prior run on the server. Never overwrite or delete a previous run because another run uses the same folder name.
11. Saving is automatic for an active Preview Override. It works with `preview_target=node`, `sampler (VHS)`, and `both`, and without a connected browser.
12. Cancellation preserves all successfully saved previews. It does not manufacture a preview for a step whose callback never completed, nor retrospectively decode skipped steps.

Here, a run/archive is **one invocation of the wrapped sampler**. Sampling completion is sufficient to finalize its archive and enable the ZIP; downstream video decoding, muxing, or other workflow nodes may still be executing. If one preview-patched model feeds multiple samplers, each invocation receives its own archive. The node shows the most recently started invocation; all invocation archives remain on disk.

## Current implementation and constraints

| Area | Current behavior | Required change |
| --- | --- | --- |
| `minimax_preview.py::_OuterSampleWrapper.__call__` | Installs a combined sampling callback; unpacks H3 video latents and generates previews. | Own an invocation-local archive session, save selected callbacks, and finalize in all normal completion/cancellation/error paths. |
| `combined()` and `_should_skip()` | `every_n_steps`, preview cost, and animation duration can all prevent decoding a step. Even interval 1 does not currently mean every step. | Make the selected capture interval authoritative. Playback scheduling must not discard archive captures. |
| `_encode_animated_webp()` | Encodes into memory and returns base64. | Encode once to bytes for disk; derive base64 only if a retained compatibility event requires it. |
| `_send_to_node()` | Sends the current animation and metadata, without an archive/run identity. | Send persisted artifact metadata with explicit run and execution identities. |
| `finally` in wrapper | Restores the default preview method but emits no terminal event and writes no archive. | Restore the method first and finalize the archive without masking sampling exceptions. |
| `js/minimax_preview.js::buildPanel()` | One image with a status line. | Two-column preview/history panel plus archive state and download control. |
| Frontend `execution_start` listener | Resets status text only; leaves the old image in place. | Clear image, rows, selection, timers, and download state for the new execution. |
| `minimax_media.py` | Owns this repository's HTTP routes. | Add namespaced archive metadata and file-download endpoints here. |
| Node schema | Widget values serialize positionally; code explicitly requires new inputs at the end. | Append the folder-name input after `playback`; preserve all existing input names, positions, and node IDs. |

Repository guidance: follow `CONTRIBUTING.md`, keep routes under `/minimax_director*`, preserve DOM resizing behavior, and validate the feature in a real ComfyUI browser. Do not modify `pyproject.toml`, node IDs, or `LICENSE` for this feature.

## Node configuration

Append one optional string input:

```text
archive_folder_name = "MiniMaxH3-preview"
```

Its tooltip should say that this is a folder **name**, not an arbitrary server path; every sampling invocation is stored separately. Add the matching defaulted parameter to `execute()` and pass the value into the wrapper.

Continue exposing `every_n_steps`, default 1. Update its tooltip to describe saved preview cadence, actual-step filenames, and the final-step exception. Preserve defaults when loading old workflows or executing API prompts that omit the new field.

Keep `max_preview_overhead` for live presentation scheduling, but change its description honestly: it cannot cap total decode/save time while guaranteeing every selected capture. Capture takes priority. Do not silently increase the capture interval to satisfy this setting. Document this behavioral change prominently, particularly for users selecting `vae (quality)`.

The default fast decode should remain practical, but capture now adds encode/write work on every selected callback. Full VAE capture at interval 1 may be expensive. Users control that cost with `every_n_steps`, `preview_frames`, `max_resolution`, and `decode`.

## Persistent storage and naming

Resolve the output root using `folder_paths.get_output_directory()`, not the repository, working directory, or temporary directory.

Proposed layout:

```text
<ComfyUI output>/MiniMaxH3/previews/
  MyShot/
    20261025T183012123456Z_84b2a9f03c1e/
      Step001.webp
      Step006.webp
      Step011.webp
      Step030.webp
      MyShot.zip
      manifest.json
    20261025T184140654321Z_c73824e5b019/
      Step001.webp
      ...
      MyShot.zip
      manifest.json
```

The ZIP contains exactly one named folder containing the successful captures:

```text
MyShot.zip
  MyShot/
    Step001.webp
    Step006.webp
    Step011.webp
    Step030.webp
```

The tree above is illustrative; the manifest is the authoritative list of saved steps. Do not include the ZIP itself, temporary files, or the server manifest in the ZIP.

- Use a UTC timestamp plus a random identifier for `run_id`. Create its directory exclusively and retry on a collision. A timestamp or node ID alone is insufficient.
- Use the same sanitized folder name for the server grouping directory, ZIP basename, and ZIP member prefix.
- Define one shared normalizer: trim whitespace, replace path separators and forbidden/control characters, strip trailing dots/spaces, handle Windows reserved names, and limit the component length. Empty or unusable names fall back to `MiniMaxH3-preview`.
- Show the actual normalized name in the node and manifest when it differs from the input.
- Treat names as single components. Reject traversal in endpoint identifiers and check resolved paths remain under the archive root, including symlink/junction escapes. Never accept a client-supplied absolute file path.
- Write each WebP to a temporary sibling and publish it by atomic rename only after its bytes are complete. Emit its downloadable event afterward.
- Finalize ZIPs with `ZIP_STORED`: WebP is already compressed, so additional deflate work offers little benefit. Stream the saved files through `ZipFile.write()` with explicit relative member names.
- Build `<name>.zip.part`, close the ZIP successfully, then atomically rename it to `<name>.zip`. Do not advertise an unfinished ZIP as downloadable.
- No automatic retention/deletion policy in this change. Clearing the node only clears browser state. Disk usage includes both the individual WebPs and their archive copies.

## Backend design

### Archive helper

Add `minimax_preview_archive.py`, kept independent of torch and the running ComfyUI server. Inject the output root and let it own naming, storage, metadata, and ZIP assembly. This allows meaningful filesystem tests without a GPU.

Suggested interface:

```python
session = PreviewArchiveSession(output_root, folder_name, node_id, execution_context)
artifact = session.save_step(step_number, webp_bytes, preview_metadata)
result = session.finalize(sampling_status, error_summary=None)
```

`finalize()` must be idempotent. Store state on the invocation-local session, not the model wrapper: model patches may be cached, cloned, or invoked more than once.

Maintain an atomically updated `manifest.json` containing:

- Schema version, run ID, normalized/requested names, node ID, creation/finish timestamps.
- Execution/prompt and sampler identity when available from the installed runtime.
- Relevant capture settings, total steps, and last observed callback step.
- Ordered artifact records: actual step, basename, byte size, frame count, playback rate, and render cost.
- Sampling status (`running`, `completed`, `interrupted`, `failed`).
- Separate archive state (`recording`, `finalizing`, `ready`, `partial`, `unavailable`) and a concise failure reason when needed. A canceled sampling run can have a fully valid, ready ZIP of its partial captures.
- Capture failures/missing selected steps, so a valid ZIP never implies all requested captures succeeded.

Use disk metadata as the durable record; an in-memory lookup may accelerate active-run access but must not be the only way to download a retained archive after server restart.

### Capture and display separation

At wrapper entry, create a new session and emit a start event before any callback previews. Use a scope that covers decoder setup and executor execution, so setup errors also produce a terminal state when session creation succeeded.

For each callback:

1. Convert the callback's step to its actual one-based step number.
2. Record progress and determine eligibility: existing interval rule, or the last callback on successful progression. Verify the callback's `total_steps` and indexing on the supported ComfyUI runtime before relying on the last-step check.
3. For an eligible callback with usable `x0`, unpack, thin, decode, resize, and calculate playback rate using the existing H3 logic.
4. Encode one animated WebP to bytes. Persist it and update the manifest before publishing its artifact record.
5. Send a small artifact event for **every saved capture**, irrespective of live-preview target or display cadence.
6. Use a separate presentation clock to decide when to replace the left image or stream VHS frames. Preserve the existing effort to let animations finish a loop. Capture must never be skipped because a previous animation is still playing.
7. Forward the original sampling callback once, preserving its arguments and exception behavior.

Keep processing synchronous initially. It provides bounded per-step memory, avoids retaining GPU tensors, and ensures all completed saves exist when cancellation reaches finalization. Do not queue latent tensors or encode the same step twice. Consider bounded background disk work only if measurements later justify it.

For `preview_target=sampler (VHS)`, the history still receives download records, but the left node animation remains inactive according to that target. For `both`, reuse the same decoded frames for WebP and VHS output.

### Completion, cancellation, and failure

Use an explicit exception path around the executor:

- Normal return: mark sampling completed.
- The installed ComfyUI interruption exception: mark interrupted and re-raise unchanged after finalization.
- Other sampling/setup exceptions: mark failed and re-raise unchanged after finalization.
- In `finally`: restore any patched default preview method immediately, then finalize whatever files were saved and emit terminal archive state.

Inspect the actual runtime's interruption exception before implementing this branch. The current callback's broad `except Exception` must not swallow a cancellation raised during VAE decoding or capture. Interruption must escape promptly, and finalization must avoid operations that consult an already-set GPU cancellation flag.

Preview/archive errors should not turn an otherwise valid video generation into a failed render. Log and surface capture failures, retain prior files, and distinguish an incomplete capture set from a sampling failure. If storage becomes unavailable, stop repeated failing disk writes for that invocation, show the archive failure, and continue the existing live-preview behavior where possible. Never claim the save succeeded.

If ZIP assembly fails, leave individual WebPs downloadable and expose the archive failure. Do not let that exception replace the original cancellation/sampling exception. A run interrupted before any capture should still finalize a valid ZIP with an empty named directory when storage is available; label it `0 steps saved`.

Graceful ComfyUI cancellation is in scope. A process kill, power loss, or full disk cannot guarantee ZIP completion. Already published WebPs survive; a manifest still marked running after a restart should be recognized as an incomplete prior invocation. Automatic crash-recovery ZIP reconstruction can be a later enhancement and is not required for normal cancellation support.

## Events and download endpoints

Use an explicit lifecycle protocol, for example `minimax_h3_preview_archive`, with `kind` values `started`, `step_saved`, `finished`, and `archive_error`. Either extend the existing live-preview event or retain it for live presentation; do not use the artifact notification to unconditionally restart the left animation.

Every archive event should identify `node_id`, `run_id`, the current execution/prompt when available, and an increasing session-local sequence number. A saved-step event carries the actual step, filename, download/preview location, and existing frame/rate/cost metadata. A terminal event carries sampling status, saved count, capture failures, archive state, and ZIP location only when ready.

Capture the originating client/execution context at sampler entry and target notifications to that client using the supported server interface. Do not repeatedly read a mutable global client ID while saving files. If the installed runtime does not expose stable prompt identity in the wrapper, resolve that integration before implementing frontend stale-event filtering; do not guess it from the last node ID. An API/headless invocation must still archive successfully without a client recipient.

Suggested GET endpoints in `minimax_media.py`:

| Endpoint | Behavior |
| --- | --- |
| `/minimax_director/preview_archive/manifest` | Return a specified run's metadata and saved-step records. |
| `/minimax_director/preview_archive/step` | Serve a manifest-listed step as `image/webp`; support inline display and attachment download. |
| `/minimax_director/preview_archive/zip` | Serve the finalized archive as `application/zip` with attachment filename `<name>.zip`. |

Use validated folder/run identifiers and a numeric step parameter; resolve files from the manifest, not an arbitrary requested filename. Return 400 for invalid identifiers, 404 for missing artifacts, and an explicit not-ready response for a still-recording ZIP. Ensure temporary files cannot be requested.

Use aiohttp file responses so large downloads do not need to be assembled again or fully loaded into memory. Preserve the host's authentication/access conventions. Set correct content disposition and safe filename encoding. Route URLs must go through the frontend's `api.apiURL()`/`api.fetchApi()` conventions to support installations under a URL prefix.

Prefer URL-backed live images and history links over retaining a base64 copy of every animation in JavaScript. This also makes individually downloaded bytes identical to the archived bytes. Keep any transitional base64 event support limited to the current image, never the entire history.

## Frontend design and lifecycle

Extend `buildPanel()` to return references for a left preview, right history list, run-name/status line, and download-all control.

- Use a two-column flex/grid layout. Give the list its own vertical scroll area with `min-height: 0`; adding steps must not continually enlarge the node.
- Increase the initial node width enough for the right column, but preserve user resizing and the existing DOM widget minimum-height approach. Do not derive minimum size from the node's current size.
- Each row shows the actual step label/filename and a download control. Clicking the step label may select it for inspection on the left; provide a `Live` control to resume automatic display updates.
- Use lightweight text rows rather than simultaneously playing dozens of animated thumbnails.
- Keep the download-all button visible but disabled during sampling/finalization, with meaningful text. Enable it only after the server reports a ready archive. On cancellation, label the result as partial sampling, e.g. `Canceled — 12 steps saved`.
- Show the effective archive name and capture count. Use actual step numbers in the progress label and row ordering; never use the number of files as the sampler step.
- Use text nodes/`textContent` for folder names and errors. Avoid injecting user-supplied names as HTML.
- Make download controls keyboard-accessible and prevent pointer events on controls from unintentionally dragging the node.

Maintain transient panel state: execution ID, active run ID, last event sequence, artifact map keyed by step, sampling/archive status, selected step, and live-playback timer state. Do not serialize file contents, DOM elements, or transient history into the workflow.

On `execution_start`, clear the previous image and history, reset selection/download links, and cancel old display timers. On a matching `started` event, bind the new run and normalized name. Ignore late events from an older execution or invocation; deduplicate step notifications. A newly started invocation under the same prompt also resets the visible list.

Use standard execution completion/interruption/error events as reconciliation triggers, not as proof the ZIP already exists. Fetch the active run's manifest if a terminal archive event was missed. If an entirely cached run does not invoke the sampler, show `No new previews generated` rather than inventing an archive or showing stale rows. Do not disable sampler caching solely for this feature.

On reconnect, reconcile a known active run from its manifest. During in-flight sampling, a gap in event sequence should also trigger reconciliation so lost notifications do not hide saved steps. A full browser reload may start with an empty panel; retained runs remain available on disk and through known artifact URLs. A full historical-run browser is outside this change.

On node removal, clear timers/listeners owned by that node and revoke any object URLs that were created. Downloads should use normal server-backed links, avoiding a browser-side ZIP library or a giant ZIP Blob.

## Implementation sequence

1. **Verify runtime contracts.** Inspect the actual supported ComfyUI installation for callback indexing, interruption type, execution/client identity, event payloads, and URL-prefix behavior. Record the tested core/frontend versions. This repository was inspected for this plan; a live ComfyUI runtime was not verified.
2. **Implement and test the archive helper.** Establish naming, exclusive directories, atomic saves, manifests, and idempotent finalization independently of GPU code.
3. **Integrate sampler capture.** Append the folder-name widget, pass configuration, separate capture cadence from display cadence, encode once, and implement complete/interrupted/failed finalization.
4. **Add lifecycle notifications and HTTP access.** Keep route registration in `minimax_media.py`; keep the helper free of server imports to avoid circular imports and simplify tests.
5. **Build the history UI.** Add rows, selection/download links, terminal ZIP state, full new-run reset, stale-event protection, and manifest reconciliation.
6. **Update documentation.** Explain cadence examples, the final-step exception, storage paths, cancellation behavior, and the changed meaning of the overhead setting in the README. Add a changelog entry following repository conventions.
7. **Validate offline and in ComfyUI.** Complete the cases below before declaring the feature ready. Avoid unrelated workflow rewrites or registry metadata changes.

Likely files changed: `minimax_preview.py`, `js/minimax_preview.js`, `minimax_media.py`, `README.md`, `CHANGELOG.md`, and `test_node.py`; new files `minimax_preview_archive.py` and `test_preview_archive.py`. Add a focused frontend lifecycle test only if a suitable runner is available; real browser acceptance is still required.

## Validation and acceptance criteria

### Filesystem and lifecycle tests

- A 30-step callback simulation at interval 1 produces exactly `Step001.webp` through `Step030.webp`.
- Interval 5 produces steps 1, 6, 11, 16, 21, 26, and 30. Assert the actual filenames and ZIP member names; no compact renumbering.
- Step 999 and step 1000 produce `Step999.webp` and `Step1000.webp`.
- Every ZIP member's bytes equal the corresponding server WebP. Verify WebP animation decoding/frame information with a multi-frame fixture.
- Two invocations with identical folder names produce distinct directories, and the second leaves the first untouched.
- Cancellation after several callbacks preserves files, creates a readable ZIP, records interrupted status, restores the default preview method, and re-raises the same interruption.
- Cancellation during decode escapes the broad callback error handler and does not continue sampling as if nothing happened.
- Zero-capture cancellation creates the documented empty folder ZIP. Ordinary sampler failure finalizes already saved artifacts without replacing the original exception.
- Calling finalization twice does not duplicate members or corrupt the ZIP.
- Simulated encode, disk-write, manifest-write, and ZIP errors leave honest state, no published partial file, and no masked sampling error.
- Malformed names, Windows reserved names, duplicate names, traversal attempts, and paths escaping the root cannot write/read outside the configured archive area.
- Existing schema inputs retain positions/defaults, and the new schema input has a matching execute parameter.
- Stub callbacks confirm archive cadence is independent of live animation duration and `max_preview_overhead`; the original callback runs once per callback received.

### Browser and ComfyUI acceptance

- Run a short real H3 sample with fast decode and interval 1: every callback step appears on the right and individual downloads work while sampling continues.
- Change interval and confirm visible filename gaps and final-step inclusion in both disk output and ZIP.
- Cancel mid-run, including during a VAE preview, and verify ZIP readiness follows finalization. The list retains completed captures until the next execution.
- Queue another run with the same folder name: the panel clears and repopulates; prior files and ZIP remain untouched.
- Check `node`, `sampler (VHS)`, and `both`; all produce archives, and live previews still honor their selected target.
- Confirm left-image playback is not restarted by every history row insertion. Selecting an older step and returning to Live works.
- Test old saved workflows, node resizing, a long list, names with spaces/non-ASCII characters, and node removal during a run.
- Test two preview nodes, one patched model used by two samplers, two clients, and queued executions. No cross-run history mixing or notifications updating the wrong graph.
- Disconnect/reconnect while sampling and verify manifest reconciliation restores missed rows for the known run.
- Exercise a cached execution with no new sampling and an error before sampling starts; neither should advertise stale downloads as a new run.
- Verify direct downloads after restarting the server using retained run metadata, including through the deployment's URL prefix if applicable.
- Measure generation time, per-capture cost, peak memory, and disk usage for fast and VAE modes. Confirm increasing the capture interval reduces capture overhead without changing naming semantics.

Run the repository's required checks during implementation:

```text
python test_plan.py
python -m compileall -q .
node --check js/minimax_director.js
```

Also run the new focused storage tests, `node --check js/minimax_preview.js`, and `python test_node.py` using the ComfyUI interpreter/environment required by that harness. Validation performed during implementation is recorded below.

## Completion criteria

The feature is complete when selected sampler steps are durably saved with their true step numbers, each appears in a downloadable history row, successful and gracefully canceled runs produce the specified ZIP layout, and subsequent executions reset only the visible UI while retaining prior server artifacts. Completion also requires verified old-workflow compatibility and the real browser cancellation/download checks above.

## Implementation record — October 25, 2026

- Implemented archive storage, atomic WebP/ZIP publication, actual-step filenames, the final-callback exception, per-invocation directories, and manifests in `minimax_preview_archive.py`.
- Integrated capture/finalization in `minimax_preview.py`; the folder widget is appended after all existing widgets. Capture continues independently of live display scheduling. Notifications use the client and execution context captured at sampler entry.
- Added manifest and streamed file-download endpoints in `minimax_media.py`, and right-side history, downloads, selection, reset, stale-event filtering, and reconciliation in `js/minimax_preview.js`.
- Added `test_preview_archive.py` and `test_preview_ui.py`; documented their test-only dependencies in `CONTRIBUTING.md`. Runtime dependencies and registry metadata were not changed.
- Inspected upstream [sampling callback indexing](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/samplers.py), [execution context](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_execution/utils.py), and [interruption exception](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py). The wrapper resolves the prompt/sampler identity at invocation, not when the potentially cached model patch is created. This is source-contract verification, not a tested ComfyUI release claim.
- Validation: the existing planner's 371 checks and Python/JavaScript syntax checks passed. The new storage/sampler/HTTP suite and headless Chrome UI suite cover the automated cases above. Symlink tests require permission to create symlinks and skip explicitly otherwise.
- Browser tests use the real extension, real animated WebPs, and real HTTP routes under a URL prefix, with a simulated host. They exercise individual/ZIP downloads, cancellation state, new-run reset, reconnect reconciliation, stale events from multiple sampler invocations, cached executions, node removal, and 1,000-row resizing.
- Environment limitation: this checkout is not installed in ComfyUI and its Python environment has no Torch or ComfyUI `server` module. `test_node.py` cannot start here. Actual H3/VAE generation, GPU timing/memory measurements, old saved-workflow rendering in ComfyUI, and VHS integration require validation in an installed ComfyUI runtime; stub/browser-fixture results do not establish those results.
