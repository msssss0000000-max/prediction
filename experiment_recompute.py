import argparse
import csv
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def synchronize():
    torch.cuda.synchronize()


def cache_bytes(cache):
    """统计 KV cache 张量的有效字节数，不等于整个进程的显存占用。"""
    tensors = []

    # 新版本 Transformers 的 Cache。
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if isinstance(value, torch.Tensor):
                    tensors.append(value)

    # 部分旧版本的 DynamicCache。
    elif hasattr(cache, "key_cache"):
        tensors.extend(cache.key_cache)
        tensors.extend(cache.value_cache)

    # 更早版本的 tuple 格式。
    elif isinstance(cache, (tuple, list)):
        for layer in cache:
            tensors.extend(
                value for value in layer
                if isinstance(value, torch.Tensor)
            )
    else:
        raise TypeError(f"暂不支持的 cache 类型：{type(cache)}")

    if not tensors:
        raise RuntimeError("没有找到 KV 张量，请检查 Transformers 的 Cache 接口。")

    return sum(t.numel() * t.element_size() for t in tensors)


@torch.inference_mode()
def forward_step(model, input_ids, cache=None):
    output = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
    )

    # 只保留最后一个位置的 logits。
    # clone 避免切片继续引用整个 prefill logits 张量。
    next_logits = output.logits[:, -1, :].clone()
    next_cache = output.past_key_values
    del output

    return next_logits, next_cache


def make_input(tokenizer, length, device):
    """用重复文本构造指定 token 长度的输入。"""
    text = "请介绍一下计算机操作系统中的内存管理。"
    base = tokenizer.encode(text, add_special_tokens=False)

    repeats = (length + len(base) - 1) // len(base)
    ids = (base * repeats)[:length]

    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.inference_mode()
def run_case(model, tokenizer, prompt_length, decode_steps):
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()

    history = make_input(tokenizer, prompt_length, "cuda")

    # 1. 首次执行：构建输入的 KV cache。
    start = time.perf_counter()
    logits, cache = forward_step(model, history)
    synchronize()
    prefill_ms = (time.perf_counter() - start) * 1000

    # 2. 逐 token 执行。
    #
    # 此处每次采样后立即处理该 token。
    # 因此循环结束时，cache 对应整个 history。
    start = time.perf_counter()

    for _ in range(decode_steps):
        next_token = logits.argmax(dim=-1, keepdim=True)
        history = torch.cat([history, next_token], dim=1)
        logits, cache = forward_step(model, next_token, cache)

    synchronize()
    decode_ms = (time.perf_counter() - start) * 1000

    context_length = history.shape[1]
    kv_mib = cache_bytes(cache) / (1024 ** 2)

    # 记录抢占前“下一个 token”的预测，用于检查恢复正确性。
    expected_next_token = logits.argmax(dim=-1).item()

    allocated_before = torch.cuda.memory_allocated()

    # 3. 模拟重计算式抢占：
    # 删除 KV，但保留已经生成的 token 历史。
    del cache
    del logits
    gc.collect()
    synchronize()

    allocated_after = torch.cuda.memory_allocated()
    released_mib = (
        allocated_before - allocated_after
    ) / (1024 ** 2)

    # 4. 恢复：
    # 把已有历史作为完整上下文重新计算。
    start = time.perf_counter()
    logits, cache = forward_step(model, history)
    synchronize()
    recovery_ms = (time.perf_counter() - start) * 1000

    actual_next_token = logits.argmax(dim=-1).item()
    next_token_match = expected_next_token == actual_next_token

    # 5. 恢复后继续执行一个 token，确认链条可运行。
    next_token = logits.argmax(dim=-1, keepdim=True)
    logits, cache = forward_step(model, next_token, cache)
    synchronize()

    result = {
        "prompt_tokens": prompt_length,
        "decode_steps": decode_steps,
        "context_tokens_at_preempt": context_length,
        "kv_mib": round(kv_mib, 3),
        "released_allocated_mib": round(released_mib, 3),
        "prefill_ms": round(prefill_ms, 3),
        "decode_total_ms": round(decode_ms, 3),
        "decode_step_mean_ms": round(
            decode_ms / decode_steps, 3
        ),
        "recovery_ms": round(recovery_ms, 3),
        "next_token_match": next_token_match,
    }

    del cache, logits, history
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B"
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+",
        default=[128, 512, 1024],
    )
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="recompute_results.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("此实验需要可用的 CUDA GPU。")
    if args.decode_steps <= 0 or args.repeats <= 0:
        raise ValueError("decode-steps 和 repeats 必须大于 0。")
    if any(length <= 0 for length in args.lengths):
        raise ValueError("输入长度必须大于 0。")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda").eval()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)

    # 预热，不计入正式结果。
    run_case(model, tokenizer, 32, 4)

    rows = []
    for length in args.lengths:
        for repeat in range(args.repeats):
            row = run_case(
                model, tokenizer, length, args.decode_steps
            )
            row["repeat"] = repeat
            rows.append(row)
            print(row)

    with open(args.output, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("结果已保存到:", args.output)


if __name__ == "__main__":
    main()