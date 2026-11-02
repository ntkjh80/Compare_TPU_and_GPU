"""동일 JAX 수식으로 실행하는 Qwen2 Dense 진단 구현.

Prefill, batch=1, padding 없음, KV cache 없음. 운영 vLLM 구현은 아닙니다.
모든 행렬곱은 FP32 누산을 요청한 뒤 지정된 활성 dtype으로 명시적으로 변환합니다.
정규화 통계와 softmax는 FP32, 정규화 출력/attention 확률은 활성 dtype입니다.
"""
from functools import partial
import jax
import jax.numpy as jnp


def dtype_for(name):
    return {"bf16": jnp.bfloat16, "fp32": jnp.float32}[name]


def dot(a, b, precision):
    # precision은 FP32 피연산자의 내부 정밀도에도 영향을 줍니다.
    return jnp.matmul(a, b, precision=precision, preferred_element_type=jnp.float32)


def linear(x, w, b, precision):
    # 이 진단 구현은 matmul을 먼저 반올림한 뒤 bias를 더합니다.
    # 운영 라이브러리의 fused addmm와 반올림 경계까지 같다고 가정하지 않습니다.
    y = dot(x, w.T, precision).astype(x.dtype)
    return y if b is None else (y + b).astype(x.dtype)


def norm(x, w, eps):
    f = x.astype(jnp.float32)
    normalized = f * jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps)
    return (normalized.astype(x.dtype) * w).astype(x.dtype)


def rope(x, positions, theta):
    dim = x.shape[-1]
    inv = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freq = positions.astype(jnp.float32)[:, None] * inv[None, :]
    emb = jnp.concatenate([freq, freq], axis=-1)[None, None, :, :]
    cos, sin = jnp.cos(emb).astype(x.dtype), jnp.sin(emb).astype(x.dtype)
    rotated = jnp.concatenate([-x[..., dim // 2:], x[..., :dim // 2]], axis=-1)
    return (x * cos + rotated * sin).astype(x.dtype)


def block(x, w, positions, config, precision="highest", capture=False):
    """블록 하나. capture=False와 True를 각각 실행해 계측 영향도 검사합니다."""
    eps = config["rms_norm_eps"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    b, t, hidden = x.shape
    d = hidden // heads
    stages = {"input": x}
    h = norm(x, w["input_layernorm.weight"], eps)
    stages["input_norm"] = h
    qs = []
    for name, count in [("q", heads), ("k", kv), ("v", kv)]:
        key = f"self_attn.{name}_proj"
        y = linear(h, w[key + ".weight"], w.get(key + ".bias"), precision)
        stages[name + "_proj"] = y
        qs.append(y.reshape(b, t, count, d).transpose(0, 2, 1, 3))
    q, k, v = qs
    q, k = rope(q, positions, config["rope_theta"]), rope(k, positions, config["rope_theta"])
    stages["q_rope"], stages["k_rope"] = q, k
    k, v = jnp.repeat(k, heads // kv, axis=1), jnp.repeat(v, heads // kv, axis=1)
    scores = dot(q, k.swapaxes(-1, -2), precision).astype(x.dtype)
    scores = (scores * (d ** -0.5)).astype(x.dtype)
    # 저장하는 scores는 mask 적용 전의 유한 값입니다.
    stages["scores_unmasked"] = scores
    causal = jnp.arange(t)[:, None] >= jnp.arange(t)[None, :]
    masked = jnp.where(causal, scores.astype(jnp.float32), -jnp.inf)
    probs = jax.nn.softmax(masked, axis=-1).astype(x.dtype)
    stages["attention_probs"] = probs
    context = dot(probs, v, precision).astype(x.dtype).transpose(0, 2, 1, 3).reshape(b, t, hidden)
    stages["context"] = context
    attn = linear(context, w["self_attn.o_proj.weight"], w.get("self_attn.o_proj.bias"), precision)
    stages["attention_output"] = attn
    residual = (x + attn).astype(x.dtype)
    stages["attention_residual"] = residual
    h = norm(residual, w["post_attention_layernorm.weight"], eps)
    stages["post_attention_norm"] = h
    gate = linear(h, w["mlp.gate_proj.weight"], w.get("mlp.gate_proj.bias"), precision)
    up = linear(h, w["mlp.up_proj.weight"], w.get("mlp.up_proj.bias"), precision)
    stages["gate_proj"], stages["up_proj"] = gate, up
    activated = jax.nn.silu(gate).astype(x.dtype)
    product = (activated * up).astype(x.dtype)
    stages["silu_gate"], stages["gated_product"] = activated, product
    down = linear(product, w["mlp.down_proj.weight"], w.get("mlp.down_proj.bias"), precision)
    stages["down_proj"] = down
    out = (residual + down).astype(x.dtype)
    stages["output"] = out
    return (out, stages) if capture else out


def block_weights(weights, layer):
    prefix = f"model.layers.{layer}."
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}


def get_block(config, precision, capture):
    # 설정은 함수 안에 고정하고 입력/가중치는 JIT 인자로 전달합니다.
    return jax.jit(partial(block, config=config, precision=precision, capture=capture))


def logits(x, weights, config, precision):
    x = norm(x, weights["model.norm.weight"], config["rms_norm_eps"])
    head = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return linear(x, head, None, precision).astype(jnp.float32)
