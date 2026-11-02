# LeVo model and deployment work

For model replacement, use the [adaptation skill](skills/levo-model-adaptation/SKILL.md), [deployment guide](docs/MODEL_DEPLOYMENT.md) and [current summary](docs/PROJECT_SUMMARY.md). User instructions take precedence over this guide.

- Model profiles and source jobs belong in `configs/`; conversion/packaging in `levo_harmony/deployment/`; native inference in `runtime/include/levo/`. App code handles UI, tasks, model selection and playback. Legacy exporters/benchmarks remain reproducible interfaces, not the default deployment entry.
- The public primary Mixed LM is trained; the current0.6B music adapters and small DiT are untrained. Shape compatibility, finite PCM or a training-status declaration does not prove music quality. Do not replace missing weights with random tensors to make a deployment pass.
- Model bundles are immutable versions with a new `modelId`. Preserve the original model, shared Codec identity and old experiments. New incompatible conditioning/Codec shapes need an adapter rather than silent reshaping or changed token offsets.
- Product UI uses model names and “歌曲模型”, keeping research configuration in reports/the research App. Keep generation/import model switching disabled.
- Run `python -m unittest discover -s tests -v` for Python behavior changes, build affected Apps with `build.ps1 -WithMnn` or `tools/build_music_app.ps1`, and validate actual selected-bundle generation for runtime changes. Pure documentation updates do not need phone inference.
- Report effective audio duration, stage/whole RTF, requested recipe, HAP/model hashes and memory scope. Record background/asleep periods. Fresh module tests and recovered phase sums are not single-run E2E results; RSS/PSS are not dedicated VRAM.
- Maintain the single `reports/levo-test-report.json`, preserving historical sections. Keep weights/audio/local SDK configuration/signing material out of commits; use explicit paths when staging. Publish only within the user's authorized scope.
