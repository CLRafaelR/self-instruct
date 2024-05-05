import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from llama_cpp import Llama
import json
import time
import gc
import os

torch.random.manual_seed(0)

model_name = "microsoft/Phi-3-mini-4k-instruct-gguf"

hf_hub_download(
    repo_id=model_name,
    filename="Phi-3-mini-4k-instruct-q4.gguf",
)


model = Llama(
    model_path=f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/c80d904a71b99a3eaeb8d3dbf164166384c09dc3/Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
    n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=34,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=100,  # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)


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

start_time = time.time()

output = model(
    f"<|user|>\n{user_prompt}<|end|>\n<|assistant|>",
    max_tokens=500,  # Generate up to 1500 tokens
    stop=["<|end|>"],
    echo=False,  # Whether to echo the prompt
)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

end_time = time.time()

exec_time = end_time - start_time

print(
    output["choices"][0]["text"],
    f"###\n\n{exec_time} sec elapsed.\n\n###",
    sep="\n\n",
)
