"""공식 모델 다운로드와 공유 입력 생성. 한 번 생성한 bundle을 양쪽에 복사합니다."""
import re
import numpy as np
from pathlib import Path
from .io import read_json, write_json, new_dir, safetensor_arrays, validate_config, make_manifest, sha256


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    # main을 두 번 독립적으로 받지 않고, 먼저 불변 commit SHA를 확정합니다.
    revision = HfApi().model_info(args.model_id, revision=args.revision).sha
    dst = new_dir(args.out)
    snapshot_download(args.model_id, revision=revision, local_dir=dst,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*"])
    write_json(dst / "download_identity.json", {"repo_id": args.model_id, "revision": revision})
    print(f"다운로드 완료: {dst}\nrevision: {revision}")


def prepare(args):
    from tokenizers import Tokenizer
    src, out = Path(args.model_dir), new_dir(args.out)
    c = read_json(src / "config.json")
    validate_config(c)
    if not 1 <= args.seq_len <= c["max_position_embeddings"]:
        raise ValueError("seq-len 범위 오류")
    (out / "weights").mkdir()
    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise ValueError("모델 디렉터리에 safetensors 파일이 없습니다.")
    keys = set()
    for file in files:
        for key, value in safetensor_arrays(file):
            if key in keys or not re.fullmatch(r"[A-Za-z0-9_.]+", key):
                raise ValueError(f"중복 또는 잘못된 가중치 이름: {key}")
            keys.add(key)
            np.save(out / "weights" / (key + ".npy"), value, allow_pickle=False)
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(c["num_hidden_layers"]):
        required |= {f"model.layers.{i}.{suffix}" for suffix in [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]}
    if not c.get("tie_word_embeddings"):
        required.add("lm_head.weight")
    if not required <= keys:
        raise ValueError(f"필수 가중치 누락: {sorted(required-keys)}")
    tokenizer = Tokenizer.from_file(str(src / "tokenizer.json"))
    (out / "inputs").mkdir()
    prompts = []
    for line in Path(args.prompts).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = __import__("json").loads(line)
        name = p["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in [x["id"] for x in prompts]:
            raise ValueError("prompt ID는 중복되지 않는 영문/숫자/_/-여야 합니다.")
        ids = tokenizer.encode(p["text"], add_special_tokens=False).ids
        if not ids:
            raise ValueError(f"빈 prompt: {name}")
        original = len(ids)
        if args.fill == "repeat":
            ids = (ids * ((args.seq_len + len(ids) - 1) // len(ids)))[:args.seq_len]
        else:
            ids = ids[:args.seq_len]
        np.save(out / "inputs" / f"{name}.npy", np.asarray([ids], dtype=np.int32), allow_pickle=False)
        prompts.append({**p, "original_tokens": original, "used_tokens": len(ids)})
    if not prompts:
        raise ValueError("입력 prompt가 없습니다.")
    write_json(out / "config.json", c)
    write_json(out / "prompts.json", prompts)
    identity = read_json(src / "download_identity.json") if (src / "download_identity.json").exists() else {"local_checkpoint": True}
    make_manifest(out, {"source": identity, "source_files": {p.name: sha256(p) for p in files},
                        "tokenizer_sha256": sha256(src / "tokenizer.json"), "fill": args.fill,
                        "seq_len": args.seq_len, "weight_storage": "fp32 exact expansion of checkpoint values"})
    print(f"공통 bundle 생성 완료: {out}")
