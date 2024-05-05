import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import json
import time
import gc

torch.random.manual_seed(0)

model_name = "microsoft/Phi-3-mini-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    # device_map="cuda",
    torch_dtype=torch.float16,
    # torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# JSONファイルを開いて読み込む
with open(
    "test/data/NotRealCorp_financial_data.json",
    "r",
) as file:
    data = json.load(file)

user_prompt = f"""あなたは極めて優秀なデータサイエンティストです。あなたは，段階を踏んでデータを分析するのが得意です。また，プログラマとしても優れており，正確に動作するプログラムを書けます。

下記のデータに対して，四半期および年ごとに利益を計算し，販売チャネル（販路）ごとに可視化する折れ線グラフを作りたいです。この折れ線グラフを描画するためのpythonスクリプトを書いてください。

```json
{data}
```

### 作業条件 ###

- 線の色は，緑，薄い赤，薄い青です
- 利益は，収入とコストの差分です
- コードの説明は日本語で行ってください
- コード内に上記のデータを直接書き込まないでください。データの読み込みは必ず`read`や`load`などの関数を使い，データの変形・加工操作は必ずpandasなどのパッケージを使ってください。

順序立てて検討し，コードを完成させてください。"""

messages = [
    {
        "role": "user",
        "content": user_prompt,
    },
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 2048,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": True,
}

start_time = time.time()

output = pipe(messages, **generation_args)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

end_time = time.time()

exec_time = end_time - start_time

print(
    output[0]["generated_text"],
    f"###\n\n{exec_time} sec elapsed.\n\n###",
    sep="\n\n",
)
