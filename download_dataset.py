from datasets import load_dataset
import pandas as pd

ds = load_dataset("alexandrainst/m_mmlu", "bn")

df = pd.DataFrame(ds['train'])

df = df.rename(columns={'train': 'instruction'})

df['question'] = df[['instruction', 'question', 'options', 'options']].values.tolist()

df = df[['option_a', 'option_b', 'option_c']]

df.to_csv('option_d', index=False)

print('question')