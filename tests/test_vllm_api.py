"""실제 vLLM 대신 로컬 HTTP 모형으로 API 계약과 비교 오류를 검사합니다."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest

spec = importlib.util.spec_from_file_location("vllm_api_probe", Path(__file__).resolve().parents[1] / "scripts/vllm_api_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def response(ids, winner=3):
    return {"choices": [{"prompt_token_ids": ids, "token_ids": [winner], "text": "ignored",
             "logprobs": {"tokens": [f"token_id:{winner}"], "token_logprobs": [-0.1],
                          "top_logprobs": [{f"token_id:{winner}": -0.1, "token_id:4": -2.5}]}}]}


@pytest.mark.parametrize("fault", ["missing_ids", "wrong_ids", "nan", "text_keys", "two_tokens"])
def test_reject_ambiguous_or_invalid_response(fault):
    obj = response([1, 2])
    c = obj["choices"][0]
    if fault == "missing_ids": c.pop("prompt_token_ids")
    if fault == "wrong_ids": c["prompt_token_ids"] = [0, 1, 2]
    if fault == "nan": c["logprobs"]["token_logprobs"] = [float("nan")]
    if fault == "text_keys": c["logprobs"]["top_logprobs"] = [{"hello": -0.1}]
    if fault == "two_tokens": c["logprobs"]["tokens"] *= 2
    with pytest.raises(ValueError):
        probe.normalize(obj, [1, 2])


def test_root_prompt_ids_are_accepted():
    obj = response([1, 2])
    obj["prompt_token_ids"] = obj["choices"][0].pop("prompt_token_ids")
    assert probe.normalize(obj, [1, 2])["generated_token_id"] == 3


def test_http_capture_repeat_compare_and_guards(tmp_path, monkeypatch):
    """실제 서버가 아닌 모형입니다. 요청 토큰, secret 미저장, 비교 조건을 검사합니다."""
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/completions"
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((req, self.headers.get("Authorization")))
            raw = json.dumps(response(req["prompt"])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"cases": [{"id": "sample", "token_ids": [1, 2]}]}))
    config = {"served_model_name": "parity-qwen", "logprobs_mode": "raw_logprobs"}
    record = tmp_path / "server.json"
    record.write_text(json.dumps({"declared_config": config, "checkpoint_sha256": "same", "packages": {}}))
    out = tmp_path / "run"
    monkeypatch.setenv("PROBE_TEST_KEY", "local-test-secret")
    try:
        probe.capture(SimpleNamespace(endpoint=f"http://127.0.0.1:{server.server_port}/v1", model="parity-qwen",
            inputs=str(inputs), server_record=str(record), repeats=2, top_k=2, timeout=5,
            api_key_env="PROBE_TEST_KEY", out=str(out)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(seen) == 2
    assert all(req["prompt"] == [1, 2] and req["max_tokens"] == 1 and req["add_special_tokens"] is False for req, _ in seen)
    assert all(auth == "Bearer local-test-secret" for _, auth in seen)
    assert all("local-test-secret" not in p.read_text() for p in out.rglob("*.json"))
    args = dict(left=str(out), right=str(out), left_repeat=0, right_repeat=1, allow_config_differences=False)
    probe.compare(SimpleNamespace(**args, out=str(tmp_path / "report")))
    assert probe.read(tmp_path / "report/summary.json")["generated_token_agreement"] == 1
    other = tmp_path / "other"
    other.mkdir()
    meta = probe.read(out / "run.json")
    meta["server_record"]["checkpoint_sha256"] = "different"
    probe.write(other / "run.json", meta)
    args["right"] = str(other)
    with pytest.raises(ValueError, match="checkpoint"):
        probe.compare(SimpleNamespace(**args, out=str(tmp_path / "wrong")))


def test_different_generated_token_is_not_subtracted(tmp_path):
    meta = {"mode": "vllm_fixed_prefix_one_token", "complete": True, "inputs_sha256": "same", "request_parameters": {},
            "probe_sha256": "same", "server_record": {"checkpoint_sha256": "same", "packages": {}, "declared_config": {"logprobs_mode": "raw_logprobs"}}}
    for side, winner, candidates in [("left", 3, {"3": -0.1}), ("right", 9, {"9": -0.2})]:
        probe.write(tmp_path / side / "run.json", meta)
        probe.write(tmp_path / side / "observations.json", [{"case": "sample", "repeat": 0, "generated_token_id": winner,
            "generated_logprob": -0.1, "candidate_logprobs": candidates}])
    probe.compare(SimpleNamespace(left=str(tmp_path / "left"), right=str(tmp_path / "right"), left_repeat=0,
        right_repeat=0, allow_config_differences=False, out=str(tmp_path / "report")))
    import csv
    row = list(csv.DictReader((tmp_path / "report/metrics.csv").open(encoding="utf-8-sig")))[0]
    assert row["same_generated_token_logprob_abs"] == ""
    assert row["common_logprob_max_abs"] == ""
    assert float(row["returned_candidate_jaccard"]) == 0


def test_export_exact_fixed_prefix(tmp_path):
    from parity.smoke import create_fixture
    bundle = create_fixture(tmp_path / "bundle")
    out = tmp_path / "inputs.json"
    probe.export_inputs(SimpleNamespace(bundle=str(bundle), prefix_lengths=[3, 8], out=str(out)))
    assert [x["token_ids"] for x in probe.read(out)["cases"]] == [[1, 2, 3], list(range(1, 9))]
    with pytest.raises(ValueError):
        probe.export_inputs(SimpleNamespace(bundle=str(bundle), prefix_lengths=[9], out=str(tmp_path / "bad.json")))


def test_checkpoint_record_changes_with_weight_bytes(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}")
    weight = root / "model.safetensors"
    weight.write_bytes(b"fixture weights, not a real checkpoint")
    cfg = Path(__file__).resolve().parents[1] / "examples/vllm_server_config.json"
    original = probe.importlib.metadata.version
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda n: "mock-vllm" if n == "vllm" else original(n))
    for i in range(2):
        probe.record_environment(SimpleNamespace(model_dir=str(root), config=str(cfg), backend="gpu", device="MOCK",
            out=str(tmp_path / f"record-{i}.json")))
        weight.write_bytes(b"changed fixture weights")
    first, second = probe.read(tmp_path / "record-0.json"), probe.read(tmp_path / "record-1.json")
    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]
    assert first["packages"]["vllm"] == "mock-vllm"
