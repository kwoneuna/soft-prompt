from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

event_path = "/workspace/Soft-Prompt-Generation/outputs_baseline/real_dongwon/CoOp/office_home/b32_ep50/ViT-B16/a/seed_1/warmup_1/ana/tensorboard_log/events.out.tfevents.1762848892.95d55bbdfefd.104196.1"   # 이벤트 파일 경로

ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

all_scalars = {}

for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    df = pd.DataFrame({'step': steps, tag: values})
    all_scalars[tag] = df

# merge all tags
result = None
for tag, df in all_scalars.items():
    if result is None:
        result = df
    else:
        result = result.merge(df, on='step', how='outer')

result.to_csv("tensorboard_export.csv", index=False)
print("Saved as tensorboard_export.csv")
