import torch
from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)
from transformers import (
    AutoTokenizer,
    AutoConfig,
)
import json
import time
import gc
from optimum.pipelines import pipeline
import onnxruntime
from optimum.onnxruntime import (
    ORTModelForQuestionAnswering,
    ORTModelForCausalLM,
)
from optimum.exporters import TasksManager
from optimum.utils import NormalizedConfigManager
import os

torch.random.manual_seed(0)

model_name = "microsoft/Phi-3-mini-128k-instruct-onnx"
provider_name = model_name.split("/", 1)[0]
model_name_short = model_name.split("/", 1)[1]

file_name = "cuda/cuda-fp16/phi3-mini-128k-instruct-cuda-fp16.onnx"

"""
snapshot_download(
    repo_id=model_name,
)
"""

session_options = onnxruntime.SessionOptions()
# session_options.log_severity_level = 0
session_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
)

config = AutoConfig.from_pretrained(
    f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--{provider_name}--{model_name_short}/snapshots/e85d1b352b6f6f2a30d188f35c478af323af2449/cuda/cuda-fp16",
    force_download=False,
    trust_remote_code=True,
)

# Copy the settings for `phi` and add them as the settings for `phi3`
# https://github.com/huggingface/optimum/issues/1826#issuecomment-2075070853
TasksManager._SUPPORTED_MODEL_TYPE["phi3"] = TasksManager._SUPPORTED_MODEL_TYPE["phi"]
NormalizedConfigManager._conf["phi3"] = NormalizedConfigManager._conf["phi"]


model = ORTModelForCausalLM.from_pretrained(
    f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--{provider_name}--{model_name_short}/snapshots/e85d1b352b6f6f2a30d188f35c478af323af2449/cuda/cuda-fp16",
    # provider="CPUExecutionProvider",
    provider="CUDAExecutionProvider",
    trust_remote_code=True,
    local_files_only=True,
    config=config,
    force_download=False,
    session_options=session_options,
    use_io_binding=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--{provider_name}--{model_name_short}/snapshots/e85d1b352b6f6f2a30d188f35c478af323af2449/cuda/cuda-fp16"
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

user_prompt = "Hi!"

messages = [
    {
        "role": "user",
        "content": user_prompt,
    },
]


pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    # accelerator="ort",
    device="cuda:0",
    torch_dtype=torch.float16,
)

generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "temperature": 0.01,
    "do_sample": True,
}

start_time = time.time()

output = pipe(
    user_prompt,
    # question=user_prompt,
    # context="あなたは日本語で返答します",
    **generation_args,
)

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
