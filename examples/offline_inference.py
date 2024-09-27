import torch
from torch.profiler import ExecutionTraceObserver, profile

from vllm import LLM, SamplingParams

def trace_handler(prof):
    prof.export_chrome_trace("kineto_trace.json")

# Sample prompts.
prompts = [
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)

# Create an LLM.
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B", max_model_len=18000)

et = ExecutionTraceObserver()
et.register_callback("pytorch_et.json")
et.start()

with profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=10, active=1),
    on_trace_ready=trace_handler
) as prof:
    outputs = llm.generate(prompts, sampling_params)

et.stop()
et.cleanup()

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")