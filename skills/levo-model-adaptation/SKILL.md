---
name: levo-model-adaptation
description: Adapt replacement LeVo Mixed or 0.6B music-student weights into validated MNN bundles, switch models in the Harmony apps, and test deployment. Use for checkpoint, TorchScript, ONNX or MNN replacement in this repository, not generic model training.
---

Use the model toolchain and bundle interfaces rather than editing App filenames, compiled hashes or scattered historical exporters.

Locate the repository containing `tools/modelctl.py` and `configs/profiles/`. The current user's clone is `C:/levo-harmony`; other machines may use another path. Read `docs/MODEL_DEPLOYMENT.md` for commands and `docs/PROJECT_SUMMARY.md` for measured status. For format/shape decisions read [contracts](references/contracts.md).

## Select the adapter

- `public-mixed-v2`: trained primary Mixed LM,39-layer music DiT,shared Oobleck decoder. The Music App supports this conditioning interface.
- `student-mixed-v1`:26-layer Qwen3 music student,22-layer smaller music-interface DiT,same decoder. The current student music adapters/DiT are untrained; use LeVo Harmony for structure/performance tests.
- Match codebook, normalization, condition tables, tokenizer and embedding rows, not only hidden size. A changed Codec contract needs a runtime adapter. A text checkpoint alone is not a trained music checkpoint.

## Prepare and package

1. Inspect with `modelctl inspect`. Identify component,format and training status. PT/PTH/Torch state dictionaries need architecture config and explicit `stateKey`/`stripPrefix` where appropriate. Do not use `strict=False`, invent missing parameters, or enable pickle fallback to obtain a successful load.
2. Create a job using `configs/deployment/` examples. Run `modelctl prepare --job ... --out ...`. Use the matching environment: legacy upstream GPT dependencies for DiT,modern Transformers for LM. The isolated MNN converter extension must match interpreter version and bitness.
3. Preserve failed conversions and measured errors. Do not quietly widen a tolerance until a model passes. State any changed numerical budget and its baseline evidence; numerical parity does not prove listening quality.
4. `modelctl pack` takes a base bundle and replaced components. Use a new `modelId` and concise `displayName`; never overwrite the previous package. `modelctl verify` must pass before transfer.
5. Check `component.json`/`model-bundle.json`: distinguish declared weight bits,requested runtime precision/threads,numerical validation and training provenance. W4 storage does not imply A4/KV4. ONNX I/O alone does not identify internal layer count.

## Deploy and validate

- `modelctl deploy ... --app music` installs an independent public trained bundle. In LeVo 音乐 refresh and choose it under **歌曲模型**. The menu uses the same `models`/`selectModel` native APIs. Generation/import disables switching.
- `modelctl deploy ... --app research` keeps student/experimental bundles for `run_comparison_app.ps1 -ModelBundle <id>`; it does not globally switch the product.
- Test default→replacement→default,an unavailable ID,and a short real generation. Check `modelBundle`,graph/weight hashes,finite PCM,actual WAV duration and playback callback. Copied files/build success do not establish generation success.
- Keep existing phone work intact. Check whether a task is active before force-stopping an App. Do not restart pending inference just because logs pause. Ask for foreground action if needed and continue independent work; mark incomplete observation rather than presenting it as a controlled timing baseline.
- Record effective audio length,stage/whole RTF,peak RSS/PSS where measured,configuration,HAP hash and foreground scope. Recovered phase sums are not uninterrupted end-to-end RTF. A normal exit alone does not prove OOM.

Keep experiments in `experiments/catalog.json`,outside the default pipeline. Preserve conclusions and reproducible entries without adding every diagnostic to product UI.

Publish source,configuration,skill,docs and the unified report when repository publication is authorized. Keep weights,audio,signing files and raw device logs out of Git. Preserve enough provenance for another Agent to repeat the adapter instead of guessing from a filename.
