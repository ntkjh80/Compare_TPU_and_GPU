"""단일 JAX device 실행. GPU/TPU 요청 시 CPU fallback을 허용하지 않습니다."""
import os
from pathlib import Path
import numpy as np
from .io import read_json, write_json, sha256, new_dir, verify_bundle, load_weights, validate_config, environment
from .metrics import metrics


def setup(args):
    # JAX import 전에 backend를 강제해 가속기 초기화 실패를 숨기지 않습니다.
    os.environ["JAX_PLATFORMS"] = {"gpu": "cuda", "tpu": "tpu", "cpu": "cpu"}[args.backend]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_default_matmul_precision", args.precision)
    devices = jax.local_devices()
    if not devices or any(d.platform != args.backend for d in devices):
        raise RuntimeError(f"요청 backend와 감지 장비 불일치: {devices}")
    if not 0 <= args.device_index < len(devices):
        raise ValueError("device-index 범위 오류")
    device = devices[args.device_index]
    jax.config.update("jax_default_device", device)
    return jax, device


def meta(args, bundle_id, device, jax):
    safe_env = {k: os.environ[k] for k in ["JAX_PLATFORMS", "XLA_FLAGS", "LIBTPU_INIT_ARGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "CUDA_VISIBLE_DEVICES"] if k in os.environ}
    source = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    return {"schema": 1, "bundle_sha256": bundle_id, "dtype": args.dtype, "precision": args.precision,
            "backend": args.backend, "device": str(device), "device_kind": device.device_kind,
            "device_index": args.device_index, "local_devices": [str(d) for d in jax.local_devices()],
            "environment": environment(), "flags": safe_env, "source_sha256": source,
            "operator_policy": "fp32 accum request; activation casts; fp32 norm statistics and softmax",
            "scope": "single-device prefill, no KV cache, no sharding, batch=1"}


def save_array(path, value, jax):
    np.save(path, np.asarray(jax.device_get(value), dtype=np.float32), allow_pickle=False)


def device_weights(host, dtype, device, jax):
    import jax.numpy as jnp
    return {k: jax.device_put(jnp.asarray(v, dtype=dtype), device) for k, v in host.items()}


def run(args):
    root = Path(args.bundle)
    identity = verify_bundle(root)
    c = read_json(root / "config.json")
    validate_config(c)
    out = new_dir(args.out)
    jax, device = setup(args)
    from .model import dtype_for, block_weights, get_block, logits
    dtype = dtype_for(args.dtype)
    host = load_weights(root)
    w = device_weights(host, dtype, device, jax)
    plain, traced = get_block(c, args.precision, False), get_block(c, args.precision, True)
    head_weights = {k: w[k] for k in ["model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"] if k in w}
    head = jax.jit(lambda x, hw: logits(x, hw, c, args.precision))
    manifest = meta(args, identity, device, jax)
    manifest.update({"mode": "full", "capture_layers": sorted(args.capture_layers), "repeats": args.repeats})
    if any(i < 0 or i >= c["num_hidden_layers"] for i in args.capture_layers):
        raise ValueError("capture-layers 범위 오류")
    checks = []
    import jax.numpy as jnp
    for inp in sorted((root / "inputs").glob("[!.]*.npy")):
        ids = np.load(inp, allow_pickle=False)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.min() < 0 or ids.max() >= c["vocab_size"]:
            raise ValueError("잘못된 token ID 형태/범위")
        positions = jax.device_put(jnp.arange(ids.shape[1], dtype=jnp.int32), device)
        for repeat in range(args.repeats):
            dst = out / f"repeat_{repeat:02d}" / inp.stem
            dst.mkdir(parents=True)
            x = w["model.embed_tokens.weight"][ids]
            for layer in range(c["num_hidden_layers"]):
                prefix = f"layer_{layer:03d}"
                save_array(dst / f"{prefix}.input.npy", x, jax)
                bw = block_weights(w, layer)
                # 본 실행은 항상 동일한 '출력만 반환'하는 블록으로 진행합니다.
                y = plain(x, bw, positions)
                if layer in args.capture_layers:
                    yc, stages = traced(x, bw, positions)
                    check = metrics(jax.device_get(y), jax.device_get(yc))
                    checks.append({"sample": inp.stem, "repeat": repeat, "layer": layer, **check})
                    for name, array in stages.items():
                        save_array(dst / f"{prefix}.detail_{name}.npy", array, jax)
                save_array(dst / f"{prefix}.output.npy", y, jax)
                x = y
            save_array(dst / "logits.npy", head(x, head_weights), jax)
            print(f"{args.backend}: {inp.stem} repeat={repeat} 완료", flush=True)
    write_json(out / "instrumentation_checks.json", checks)
    manifest["complete"] = True
    write_json(out / "run.json", manifest)
    print(f"결과: {out}")


def replay(args):
    """다른 장비에서 가져온 하나의 실제 블록 입력을 공통으로 사용합니다."""
    root, ref = Path(args.bundle), Path(args.source_run)
    identity = verify_bundle(root)
    source = read_json(ref / "run.json")
    if not source.get("complete") or source["bundle_sha256"] != identity or source["dtype"] != args.dtype:
        raise ValueError("source-run의 완료 상태/bundle/dtype이 재실행 조건과 다릅니다.")
    c = read_json(root / "config.json")
    validate_config(c)
    if not 0 <= args.layer < c["num_hidden_layers"]:
        raise ValueError("layer 범위 오류")
    out = new_dir(args.out)
    jax, device = setup(args)
    import jax.numpy as jnp
    from .model import dtype_for, block_weights, get_block
    dtype = dtype_for(args.dtype)
    w = device_weights(block_weights(load_weights(root), args.layer), dtype, device, jax)
    plain, traced = get_block(c, args.precision, False), get_block(c, args.precision, True)
    manifest = meta(args, identity, device, jax)
    manifest.update({"mode": "replay", "layer": args.layer, "repeats": args.repeats, "replay_inputs": {}})
    checks = []
    for inp in sorted((root / "inputs").glob("[!.]*.npy")):
        source_path = ref / "repeat_00" / inp.stem / f"layer_{args.layer:03d}.input.npy"
        manifest["replay_inputs"][inp.stem] = sha256(source_path)
        x = jax.device_put(jnp.asarray(np.load(source_path, allow_pickle=False), dtype=dtype), device)
        positions = jnp.arange(x.shape[1], dtype=jnp.int32)
        for repeat in range(args.repeats):
            dst = out / f"repeat_{repeat:02d}" / inp.stem
            dst.mkdir(parents=True)
            y = plain(x, w, positions)
            yc, stages = traced(x, w, positions)
            checks.append({"sample": inp.stem, "repeat": repeat, "layer": args.layer,
                           **metrics(jax.device_get(y), jax.device_get(yc))})
            for name, array in stages.items():
                save_array(dst / f"layer_{args.layer:03d}.{name}.npy", array, jax)
            # traced output과 plain output을 별도로 보존합니다.
            save_array(dst / f"layer_{args.layer:03d}.plain_output.npy", y, jax)
    write_json(out / "instrumentation_checks.json", checks)
    manifest["complete"] = True
    write_json(out / "run.json", manifest)


def matmul(args):
    """공통 bundle의 실측 Q projection 입력과 실제 가중치로 GEMM을 분리합니다."""
    root, ref = Path(args.bundle), Path(args.source_run)
    identity = verify_bundle(root)
    source = read_json(ref / "run.json")
    if not source.get("complete") or source["bundle_sha256"] != identity or source["dtype"] != args.dtype:
        raise ValueError("source-run의 bundle/dtype/완료 상태 불일치")
    c = read_json(root / "config.json")
    validate_config(c)
    if not 0 <= args.layer < c["num_hidden_layers"]:
        raise ValueError("layer 범위 오류")
    out = new_dir(args.out)
    jax, device = setup(args)
    import jax.numpy as jnp
    from .model import dtype_for, dot
    dtype = dtype_for(args.dtype)
    weight = load_weights(root)[f"model.layers.{args.layer}.self_attn.q_proj.weight"]
    b = jax.device_put(jnp.asarray(weight.T, dtype=dtype), device)
    f = jax.jit(lambda a, b: dot(a, b, args.precision))
    manifest = meta(args, identity, device, jax)
    manifest.update({"mode": "matmul", "layer": args.layer, "repeats": args.repeats, "replay_inputs": {},
                     "scope": "Q projection GEMM only; bias excluded; FP32 accumulator output and activation-rounded output"})
    reference_rows = []
    for inp in sorted((root / "inputs").glob("[!.]*.npy")):
        name = f"layer_{args.layer:03d}.detail_input_norm.npy" if source["mode"] == "full" else f"layer_{args.layer:03d}.input_norm.npy"
        source_path = ref / "repeat_00" / inp.stem / name
        if not source_path.is_file():
            raise ValueError("input_norm이 없습니다. 해당 layer를 --capture-layers로 저장하거나 replay 결과를 사용하세요.")
        manifest["replay_inputs"][inp.stem] = sha256(source_path)
        original = np.load(source_path, allow_pickle=False)
        a = jax.device_put(jnp.asarray(original.reshape(-1, original.shape[-1]), dtype=dtype), device)
        # 장비에 들어간 실제 반올림 값을 다시 가져와 FP64 기준값을 계산합니다.
        aa, bb = np.asarray(jax.device_get(a), dtype=np.float64), np.asarray(jax.device_get(b), dtype=np.float64)
        ref64 = aa @ bb
        for repeat in range(args.repeats):
            dst = out / f"repeat_{repeat:02d}" / inp.stem
            dst.mkdir(parents=True)
            y = f(a, b)
            y_host = np.asarray(jax.device_get(y), dtype=np.float32)
            np.save(dst / "accumulator_fp32.npy", y_host, allow_pickle=False)
            save_array(dst / "rounded_activation.npy", y.astype(dtype), jax)
            np.save(dst / "reference_fp64.npy", ref64, allow_pickle=False)
            # 피연산자 해시도 비교해 입력 자체의 차이를 검출합니다.
            manifest.setdefault("operand_sha256", {})[inp.stem] = {
                "a": __import__("hashlib").sha256(aa.tobytes()).hexdigest(),
                "b": __import__("hashlib").sha256(bb.tobytes()).hexdigest()}
            reference_rows.append({"sample": inp.stem, "repeat": repeat, **metrics(y_host, ref64)})
    write_json(out / "reference_metrics.json", reference_rows)
    manifest["complete"] = True
    write_json(out / "run.json", manifest)


def doctor(args):
    jax, device = setup(args)
    import jax.numpy as jnp
    x = jax.device_put(jnp.ones((32, 32), dtype=jnp.bfloat16), device)
    result = jax.jit(lambda a: jnp.matmul(a, a, preferred_element_type=jnp.float32))(x)
    result.block_until_ready()
    if not np.all(np.asarray(result) == 32):
        raise RuntimeError("장비 matmul 검사 실패")
    data = {"requested_backend": args.backend, "selected_device": str(device), "kind": device.device_kind,
            "local_devices": [str(d) for d in jax.local_devices()], **environment()}
    print(__import__("json").dumps(data, ensure_ascii=False, indent=2))
