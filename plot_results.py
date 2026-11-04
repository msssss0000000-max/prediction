import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("recompute_results.csv")

summary = df.groupby(
    "context_tokens_at_preempt"
)[["kv_mib", "recovery_ms"]].mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(
    summary.index, summary["kv_mib"], marker="o"
)
axes[0].set_xlabel("Context tokens at preemption")
axes[0].set_ylabel("KV cache size (MiB)")
axes[0].grid(True)

axes[1].plot(
    summary.index, summary["recovery_ms"], marker="o"
)
axes[1].set_xlabel("Context tokens at preemption")
axes[1].set_ylabel("Recomputation recovery time (ms)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("recompute_results.png", dpi=200)
plt.show()