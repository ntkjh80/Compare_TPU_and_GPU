#!/usr/bin/env python3
"""vLLM의 고정 token ID 입력에 대한 1-token 응답을 수집·비교합니다.

이 파일은 레이어 tensor나 전체 logits를 추출하지 않습니다.
record/capture/compare는 Python 표준 라이브러리만 사용합니다.
export만 기존 parity 패키지와 NumPy가 필요합니다.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
from urllib import request, parse


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    """기존 결과를 덮어쓰지 않고, 비유한 숫자를 JSON으로 숨기지 않습니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def new_dir(path):
    p = Path(path)
    if p.exists() and (not p.is_dir() or any(p.iterdir())):
        raise ValueError("출력 경로가 비어 있지 않습니다.")
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_inputs(a):
    """JAX 검사와 똑같은 준비된 token ID를 표준 JSON으로 내보냅니다."""
    import numpy as np
    from parity.io import verify_bundle
    bundle = Path(a.bundle)
    bundle_id = verify_bundle(bundle)
    cases = []
    for path in sorted((bundle / "inputs").glob("[!.]*.npy")):
        ids = np.load(path, allow_pickle=False)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype.kind not in "iu":
            raise ValueError("batch=1 정수 token ID만 지원합니다.")
        lengths = a.prefix_lengths or [ids.shape[1]]
        if len(lengths) != len(set(lengths)):
            raise ValueError("prefix 길이가 중복됩니다.")
        for n in lengths:
            if not 1 <= n <= ids.shape[1]:
                raise ValueError(f"{path.stem}: prefix 길이 {n}은 입력 범위 밖입니다.")
            cases.append({"id": f"{path.stem}-prefix-{n}", "token_ids": ids[0, :n].tolist()})
    write(a.out, {"schema": 1, "bundle_sha256": bundle_id, "cases": cases})


def record_environment(a):
    """서버 venv의 패키지와 디스크 checkpoint를 기록합니다. 서버 프로세스 증명은 아닙니다."""
    root = Path(a.model_dir)
    files = sorted(set(root.glob("*.safetensors")) | set(root.glob("*.json")))
    if not list(root.glob("*.safetensors")) or not (root / "config.json").is_file():
        raise ValueError("config.json과 safetensors checkpoint가 필요합니다.")
    config = read(a.config)
    required = {"served_model_name", "dtype", "kv_cache_dtype", "logprobs_mode",
                "tensor_parallel_size", "pipeline_parallel_size", "max_model_len",
                "max_num_seqs", "prefix_caching", "chunked_prefill", "speculative_decoding"}
    if not required <= config.keys() or config["logprobs_mode"] != "raw_logprobs":
        raise ValueError("실행 설정 필수 항목과 raw_logprobs 모드를 확인하세요.")
    packages = {}
    for name in ["vllm", "vllm-tpu", "tpu-inference", "vllm-jax", "vllm-torchtpu", "vllm-torchtpu-core",
                 "torch", "torch-tpu", "tpu-sync-torch", "jax", "jaxlib", "libtpu"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if not any(k in packages for k in ["vllm", "vllm-tpu", "vllm-torchtpu"]):
        raise ValueError("vLLM 서버 패키지가 없습니다. 실제 서버의 가상환경에서 record를 실행하세요.")
    hashes = {p.name: digest(p) for p in files}
    write(a.out, {"schema": 1, "backend_label": a.backend, "device_label": a.device,
                  "python": platform.python_version(), "packages": packages,
                  "checkpoint_files": hashes, "checkpoint_sha256": identity(hashes),
                  "declared_config": config,
                  "verification_scope": "local files/packages plus operator-declared launch config; not live server attestation"})


def token_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"token_id:[0-9]+", value):
        raise ValueError("token ID 형식이 없습니다. return_tokens_as_token_ids 지원을 확인하세요.")
    return int(value.split(":", 1)[1])


def logprob(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value > 1e-6:
        raise ValueError("유한한 raw logprob가 아닙니다. raw_logits/processed 모드와 NaN/Inf를 확인하세요.")
    return float(value)


def normalize(response, expected_prompt):
    """텍스트가 아닌 실제 token ID로 정렬하고, 응답이 확인한 입력도 대조합니다."""
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("응답 choice는 정확히 하나여야 합니다.")
    c = choices[0]
    echoes = [x["prompt_token_ids"] for x in [response, c] if x.get("prompt_token_ids") is not None]
    if not echoes or any(ids != expected_prompt for ids in echoes):
        raise ValueError("prompt_token_ids 누락/불일치: 동일 입력을 검증할 수 없습니다.")
    lp = c.get("logprobs") or {}
    if len(lp.get("tokens", [])) != 1 or len(lp.get("top_logprobs", [])) != 1 or len(lp.get("token_logprobs", [])) != 1:
        raise ValueError("1개 생성 토큰의 logprobs가 필요합니다.")
    generated = token_id(lp["tokens"][0])
    if c.get("token_ids") is not None and c["token_ids"] != [generated]:
        raise ValueError("생성 token ID 필드가 서로 다릅니다.")
    candidates = {str(token_id(k)): logprob(v) for k, v in (lp["top_logprobs"][0] or {}).items()}
    if not candidates:
        raise ValueError("후보 logprobs가 비어 있습니다.")
    return {"generated_token_id": generated, "generated_logprob": logprob(lp["token_logprobs"][0]),
            "candidate_logprobs": candidates}


def post_json(endpoint, payload, key, timeout):
    """한 요청씩 보냅니다. API 키는 메모리에서만 사용하며 파일에 저장하지 않습니다."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = request.Request(endpoint + "/completions", data=json.dumps(payload).encode(), headers=headers)
    with request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def capture(a):
    parsed = parse.urlsplit(a.endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint에는 인증정보/쿼리 없는 http(s) API 주소를 사용하세요.")
    if a.repeats < 1 or a.top_k < 1 or not math.isfinite(a.timeout) or a.timeout <= 0:
        raise ValueError("repeats/top-k/timeout은 양수여야 합니다.")
    inputs, server = read(a.inputs), read(a.server_record)
    config = server["declared_config"]
    if config.get("logprobs_mode") != "raw_logprobs" or config.get("served_model_name") != a.model:
        raise ValueError("서버 기록의 모델 이름과 raw_logprobs 설정을 확인하세요.")
    cases = inputs.get("cases", [])
    names = set()
    for case in cases:
        name, ids = case["id"], case["token_ids"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("입력 ID는 안전한 영문/숫자/_/-이며 중복이 없어야 합니다.")
        if not isinstance(ids, list) or not ids or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError("비어 있지 않은 정수 token_ids가 필요합니다.")
        names.add(name)
    if not cases:
        raise ValueError("비교 입력이 없습니다.")
    out = new_dir(a.out)
    params = {"model": a.model, "max_tokens": 1, "temperature": 0, "seed": 0,
              "n": 1, "stream": False, "logprobs": a.top_k, "return_token_ids": True,
              "return_tokens_as_token_ids": True, "add_special_tokens": False,
              "top_p": 1, "top_k": -1, "min_p": 0,
              "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
              "ignore_eos": True, "skip_special_tokens": False}
    rows = []
    for repeat in range(a.repeats):
        for case in cases:
            payload = {**params, "prompt": case["token_ids"]}
            response = post_json(a.endpoint.rstrip("/"), payload, os.environ.get(a.api_key_env), a.timeout)
            row = normalize(response, case["token_ids"])
            rows.append({"case": case["id"], "repeat": repeat, **row})
            write(out / "responses" / f"{repeat:03d}-{case['id']}.json", response)
    write(out / "observations.json", rows)
    write(out / "run.json", {"schema": 1, "complete": True, "mode": "vllm_fixed_prefix_one_token",
                              "inputs_sha256": identity(inputs), "inputs": inputs,
                              "request_parameters": params, "repeats": a.repeats,
                              "server_record": server, "probe_sha256": digest(__file__),
                              "scope": "last-prefill-position top candidates; no layer tensors or cached decode"})
    print(f"수집 완료: {out}")


def compare(a):
    left, right = Path(a.left), Path(a.right)
    lm, rm = read(left / "run.json"), read(right / "run.json")
    for key in ["mode", "inputs_sha256", "request_parameters", "probe_sha256"]:
        if lm.get(key) != rm.get(key):
            raise ValueError(f"비교 조건 불일치: {key}")
    if not lm.get("complete") or not rm.get("complete"):
        raise ValueError("완료되지 않은 실행입니다.")
    ls, rs = lm["server_record"], rm["server_record"]
    if ls["checkpoint_sha256"] != rs["checkpoint_sha256"]:
        raise ValueError("checkpoint 불일치")
    changes = [k for k in set(ls["declared_config"]) | set(rs["declared_config"])
               if ls["declared_config"].get(k) != rs["declared_config"].get(k)]
    if changes and not a.allow_config_differences:
        raise ValueError(f"서버 선언 설정 불일치: {sorted(changes)}")
    if ls["declared_config"]["logprobs_mode"] != "raw_logprobs" or rs["declared_config"]["logprobs_mode"] != "raw_logprobs":
        raise ValueError("raw_logprobs끼리만 비교합니다.")
    def select(root, repeat):
        rows = [r for r in read(root / "observations.json") if r["repeat"] == repeat]
        found = {r["case"]: r for r in rows}
        if len(found) != len(rows):
            raise ValueError("중복 관측값")
        return found
    l, r = select(left, a.left_repeat), select(right, a.right_repeat)
    if not l or l.keys() != r.keys():
        raise ValueError("입력 목록/반복 번호 불일치")
    rows = []
    for case in sorted(l):
        x, y = l[case], r[case]
        p, q = x["candidate_logprobs"], y["candidate_logprobs"]
        common, union = set(p) & set(q), set(p) | set(q)
        differences = [abs(logprob(p[k]) - logprob(q[k])) for k in common]
        same = x["generated_token_id"] == y["generated_token_id"]
        rows.append({"case": case, "generated_token_agreement": int(same),
                     "left_token_id": x["generated_token_id"], "right_token_id": y["generated_token_id"],
                     "returned_candidate_jaccard": len(common) / len(union),
                     "common_candidate_count": len(common),
                     "common_logprob_max_abs": max(differences) if differences else None,
                     "common_logprob_mae": sum(differences) / len(differences) if differences else None,
                     "same_generated_token_logprob_abs": abs(logprob(x["generated_logprob"]) - logprob(y["generated_logprob"])) if same else None})
    out = new_dir(a.out)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"case_count": len(rows), "generated_token_agreement": sum(r["generated_token_agreement"] for r in rows) / len(rows),
               "config_differences": sorted(changes), "left_packages": ls["packages"], "right_packages": rs["packages"],
               "note": "공통 후보의 logprob 차이입니다. 전체 logits 오차, KL, 레이어별 행렬곱 오차가 아닙니다. 서버 설정은 운영자가 제공한 기록입니다."}
    write(out / "summary.json", summary)
    (out / "report.md").write_text("# vLLM 고정 입력 API 비교\n\n" +
        f"- 입력 수: {len(rows)}\n- 생성 token ID 일치율: {summary['generated_token_agreement']:.6f}\n" +
        f"- 선언 설정 차이: {sorted(changes)}\n\n" + summary["note"] +
        "\n\n세부 값은 metrics.csv에 있습니다. 같은 장비 반복도 별도로 비교하세요.\n", encoding="utf-8")
    print(f"보고서 생성: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("export")
    p.add_argument("--bundle", required=True)
    p.add_argument("--prefix-lengths", nargs="+", type=int)
    p.add_argument("--out", required=True)
    p = subs.add_parser("record")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--backend", choices=["gpu", "tpu"], required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--out", required=True)
    p = subs.add_parser("capture")
    p.add_argument("--endpoint", required=True, help="예: http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--server-record", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--api-key-env", default="VLLM_API_KEY")
    p.add_argument("--out", required=True)
    p = subs.add_parser("compare")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--left-repeat", type=int, default=0)
    p.add_argument("--right-repeat", type=int, default=0)
    p.add_argument("--allow-config-differences", action="store_true")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"export": export_inputs, "record": record_environment, "capture": capture, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
