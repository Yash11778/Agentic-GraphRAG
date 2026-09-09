import pandas as pd
df = pd.read_csv('eval/results/eval_results.csv')
for p in ['llm_only', 'basic_rag', 'graphrag']:
    sub = df[df['pipeline'] == p]
    pass_rate = (sub['judge'] == 'PASS').mean() * 100
    print(f'[{p}] pass={pass_rate:.1f}% avg_tokens={sub["total_tokens"].mean():.1f} avg_latency={sub["latency_s"].mean():.3f}s')
rag = df[df['pipeline'] == 'basic_rag']['total_tokens'].mean()
gr = df[df['pipeline'] == 'graphrag']['total_tokens'].mean()
print(f'Token reduction (graphrag vs basic_rag): {(1 - gr/rag)*100:.1f}%')
