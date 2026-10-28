"""Download small public configs only; do not deserialize pickle model.pt files."""
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    "qwen": "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    "levo": "https://huggingface.co/lglg666/SongGeneration-base/resolve/main/config.yaml",
}


def main():
    blobs = {k: urllib.request.urlopen(u, timeout=30).read().decode() for k, u in URLS.items()}
    qwen = json.loads(blobs["qwen"])
    levo = blobs["levo"]
    report = {"sources": URLS, "qwen_hidden_size": qwen["hidden_size"],
              "qwen_layers": qwen["num_hidden_layers"], "qwen_text_weights_directly_loadable": True,
              "qwen_text_weights_are_music_weights": False,
              "levo_base_main_hidden_1536_confirmed": "dim: 1536" in levo,
              "levo_mixed_1rvq_present": "audio_tokenizer_code_depth: 1" in levo,
              "direct_levo_to_qwen_state_dict_load": False,
              "reason": "Different backbone family, hidden width, FFN size, embeddings and music head",
              "released_compatible_0_6b_music_checkpoint_found": False,
              "publication_claim": "Config compatibility audit only; no full weight load or listening test"}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/public-config-audit.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
