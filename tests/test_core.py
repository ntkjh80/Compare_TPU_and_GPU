"""실제 수식/파일/비교 계약을 검사합니다. GPU/TPU 테스트를 대체하지 않습니다."""
import json
import struct
from types import SimpleNamespace
import numpy as np
import pytest
from parity.metrics import metrics
from parity.io import safetensor_arrays, make_manifest, verify_bundle, validate_config, write_json


def test_metrics_symmetry_zero_nonfinite():
    a = np.asarray([0., 1., -2.])
    b = np.asarray([0., 1.1, -2.1])
    assert metrics(a, a)["rel_l2"] == 0
    assert metrics(a, b)["rel_l2"] == metrics(b, a)["rel_l2"]
    assert metrics(np.zeros(2), np.zeros(2))["cosine"] == 1
    assert metrics(np.asarray([np.nan]), np.asarray([1.]))["nonfinite"] == 1
    with pytest.raises(ValueError):
        metrics(np.ones(2), np.ones(3))


def test_bf16_reader(tmp_path):
    values = np.asarray([0x3F80, 0xC020, 0x0000], dtype="<u2")
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}).encode()
    f = tmp_path / "m.safetensors"
    f.write_bytes(struct.pack("<Q", len(header)) + header + values.tobytes())
    found = dict(safetensor_arrays(f))["weight"]
    np.testing.assert_array_equal(found, [1., -2.5, 0.])


def test_manifest_detects_change(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("original")
    make_manifest(tmp_path, {})
    verify_bundle(tmp_path)
    p.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)


def test_unsupported_architecture():
    with pytest.raises(ValueError):
        validate_config({"model_type": "qwen3_moe"})


@pytest.mark.parametrize("length", [3, 11])
@pytest.mark.reference
def test_qwen_against_transformers(length):
    """별도 구현인 Transformers의 Qwen2와 FP32 block/logits를 비교합니다."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    torch.set_num_threads(1)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import jax.numpy as jnp
    from parity.model import block_weights, get_block, logits
    torch.manual_seed(17)
    cfg = Qwen2Config(vocab_size=67, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, rope_theta=1000000., tie_word_embeddings=True,
                      attention_dropout=0., use_sliding_window=False)
    cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(cfg).eval()
    weights = {k: jnp.asarray(v.detach().numpy()) for k, v in model.state_dict().items()}
    ids = np.asarray([np.arange(length) % 67], dtype=np.int64)
    captured = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def hook(m, inputs, output, index=i):
            t = output[0] if isinstance(output, tuple) else output
            captured[index] = t.detach().numpy().copy()
        handles.append(layer.register_forward_hook(hook))
    with torch.inference_mode():
        expected = model(torch.from_numpy(ids), use_cache=False).logits.numpy()
    for handle in handles:
        handle.remove()
    c = cfg.to_dict()
    fn = get_block(c, "highest", False)
    x = weights["model.embed_tokens.weight"][ids]
    for i in range(2):
        x = fn(x, block_weights(weights, i), jnp.arange(length))
        np.testing.assert_allclose(np.asarray(x), captured[i], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(logits(x, weights, c, "highest")), expected, atol=3e-6, rtol=2e-5)


def test_full_replay_matmul_and_guard(tmp_path):
    """입력 생성, 전체 레이어, 재실행, GEMM, 보고서 경로를 CPU에서 검사합니다."""
    from parity.smoke import create_fixture
    from parity.run import run, replay, matmul
    from parity.compare import compare
    bundle = create_fixture(tmp_path / "bundle")
    common = dict(bundle=str(bundle), backend="cpu", device_index=0, dtype="bf16", precision="highest", repeats=2)
    full = tmp_path / "full"
    run(SimpleNamespace(**common, out=str(full), capture_layers=[0]))
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    for dst in [r1, r2]:
        replay(SimpleNamespace(**common, out=str(dst), source_run=str(full), layer=0))
    matmul(SimpleNamespace(**common, out=str(tmp_path / "gemm"), source_run=str(full), layer=0))
    base = dict(left_repeat=0, right_repeat=1, atol=1e-5, rtol=1e-3, allow_setting_differences=False)
    compare(SimpleNamespace(**base, left=str(full), right=str(full), out=str(tmp_path / "report")))
    compare(SimpleNamespace(**base, left=str(r1), right=str(r2), out=str(tmp_path / "replay-report")))
    assert (tmp_path / "report" / "layer_difference.png").is_file()
    manifest = json.loads((r2 / "run.json").read_text())
    manifest["replay_inputs"]["synthetic"] = "wrong"
    write_json(r2 / "run.json", manifest)
    with pytest.raises(ValueError, match="replay_inputs"):
        compare(SimpleNamespace(**base, left=str(r1), right=str(r2), out=str(tmp_path / "invalid")))
