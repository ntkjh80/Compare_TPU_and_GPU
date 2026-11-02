"""다운로드 없는 CPU 실행 경로 검사. 생성 가중치는 학습된 모델이 아닙니다."""
from types import SimpleNamespace
import numpy as np
from .io import make_manifest, new_dir, write_json


def create_fixture(root):
    """작은 Qwen2 구조의 합성 bundle을 생성합니다. 테스트와 smoke가 공유합니다."""
    root = new_dir(root)
    (root / "weights").mkdir()
    (root / "inputs").mkdir()
    rng = np.random.default_rng(17)
    c = dict(model_type="qwen2", vocab_size=67, hidden_size=32, intermediate_size=48,
             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             max_position_embeddings=128, rope_theta=1000000., rms_norm_eps=1e-6,
             tie_word_embeddings=True, hidden_act="silu", use_sliding_window=False)
    weights = {"model.embed_tokens.weight": (rng.standard_normal((67, 32)) * .02).astype(np.float32),
               "model.norm.weight": np.ones(32, dtype=np.float32)}
    for i in range(2):
        prefix = f"model.layers.{i}."
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            weights[prefix + name] = np.ones(32, dtype=np.float32)
        shapes = {"self_attn.q_proj": (32, 32), "self_attn.k_proj": (16, 32),
                  "self_attn.v_proj": (16, 32), "self_attn.o_proj": (32, 32),
                  "mlp.gate_proj": (48, 32), "mlp.up_proj": (48, 32), "mlp.down_proj": (32, 48)}
        for name, shape in shapes.items():
            weights[prefix + name + ".weight"] = (rng.standard_normal(shape) * .02).astype(np.float32)
            if name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
                weights[prefix + name + ".bias"] = (rng.standard_normal(shape[0]) * .01).astype(np.float32)
    for name, array in weights.items():
        np.save(root / "weights" / (name + ".npy"), array, allow_pickle=False)
    np.save(root / "inputs" / "synthetic.npy", np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32), allow_pickle=False)
    write_json(root / "config.json", c)
    make_manifest(root, {"synthetic_test": True, "seed": 17, "trained_model": False})
    return root


def smoke(args):
    """전체 경로가 연결되는지 확인하고 사용자에게 결과 위치를 안내합니다."""
    from .run import run, replay, matmul
    from .compare import compare
    root = new_dir(args.out)
    bundle = create_fixture(root / "bundle")
    common = dict(bundle=str(bundle), backend="cpu", device_index=0, dtype="bf16", precision="highest", repeats=2)
    full = root / "full"
    run(SimpleNamespace(**common, out=str(full), capture_layers=[0]))
    replay(SimpleNamespace(**common, source_run=str(full), layer=0, out=str(root / "replay")))
    matmul(SimpleNamespace(**common, source_run=str(full), layer=0, out=str(root / "matmul")))
    compare(SimpleNamespace(left=str(full), right=str(full), left_repeat=0, right_repeat=1,
                            atol=1e-5, rtol=1e-3, allow_setting_differences=False, out=str(root / "report")))
    write_json(root / "smoke.json", {"completed": True, "backend": "cpu", "synthetic_weights": True,
              "purpose": "설치와 실행 경로 확인. GPU/TPU 정확도 검증이 아닙니다."})
    print(f"합성 입력 CPU 검사 완료: {root / 'report' / 'report.md'}")
