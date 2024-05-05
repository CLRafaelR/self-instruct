import torch
from huggingface_hub import (
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
# Run this line if you want to download the whole repository of the model
# to the default local cache for Hugging Face package
snapshot_download(
    repo_id=model_name,
)
"""

session_options = onnxruntime.SessionOptions()
# session_options.log_severity_level = 0 # The model won't be loaded if this line is executed
session_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
)

config = AutoConfig.from_pretrained(
    f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--{provider_name}--{model_name_short}/snapshots/e85d1b352b6f6f2a30d188f35c478af323af2449/cuda/cuda-fp16",
    force_download=False,
    trust_remote_code=True,
)

# The two lines below must be executed to load the model successfully!
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

user_prompt = "Hi!"

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
