# Adapters and fragile boundaries

- `levo_harmony/deployment/`: checkpoint loading,Torch adapters,conversion,bundle integrity,HDC transport.
- `runtime/include/levo/`: shared MNN session,Mixed LM,Codec,conditioning,bundle/catalog. Existing App/CLI files are compatibility facades.
- `configs/profiles/`: two current families and default recipes. Job paths are relative to their JSON; `${ROOT}` expands to repository root.

| Source | Requirements |
|---|---|
| PT/PTH/Torch/safetensors | Tensor state_dict plus architecture. Nested checkpoints use `stateKey`; explicit subtree selection uses `stripPrefix`. |
| HF LM directory | Local config,tensor shards,tokenizer;16385-row music head. |
| Student training checkpoint | `checkpointLayout: student-training`; strictly load `MixedSongStudent`,then combine text/style/music inputs and music head. Never create random missing adapters. |
| TorchScript | Explicit `format: torchscript`; compatible codec forward. Pickled Python nn.Module is not TorchScript. |
| Codec ONNX | Static I/O below; external tensor files preserved. |
| LM ONNX | MNN exporter graph: embedding-valued `input_ids`,attention_mask,position_ids,logits_index→logits. Ordinary explicit-past_key_values HF ONNX is different. Supply llm_config,tokenizer,full BF16 input table. Generic conversion is uniform W8 by default; mixed W4-body/W8-head uses HF/PT official export. |
| MNN | Adjacent external weight if used,plus LM companions. Bitwidth is declared,not inferred from filename. |

| Role | Inputs | Output |
|---|---|---|
| DiT | inputs[2,256,1200],timestep[2],attention_mask[2,1,256,256] | velocity[2,256,64] |
| VAE | latent[1,64,256] | pcm[1,2,491520] |

Codes0..16383 at25Hz,EOS16384. VAE48kHz stereo,1920samples/latent. Window256/context64/hop192. LM hidden2048/public or1024/student; Codec consumes music IDs and1024-wide code projection,not raw LM hidden.

Public conditioning requires trained text/style/structure/prompt tables. Student test prefix is512 lyric positions+style+BOS=514; offsets/row count are in diagnostic-student.json. Input embedding rows differ from16385 output classes.

CPU Low is a precision preference,not weight bit count. Current default is CPU; RSS is not dedicated VRAM. Public LM Low historically aborted during loading; retain High as its validated default. Student Low does not validate public Low.

MNN can reuse a4-thread first-created CPU pool for later6-thread requests. Requested and effective parallelism differ. Install the VAE NEON selector after backend creation and before deferred resize,restore after graph construction; fresh pre-backend installation fails.

NPU Conv probes do not prove VAE/whole-chain acceleration; an offline CANN graph may contain CPU partitions despite disabled runtime fallback. The latest small90s retest is RTF1.566/1.608. Shared VAE84.414M is identical across families. Public90s used separate-process recovery;6.225 is a phase sum,not uninterrupted E2E.
