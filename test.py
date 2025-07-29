from vllm import LLM, SamplingParams
from vllm.sampling_params import DynamicTemperatureConfig

llm = LLM(model="facebook/opt-125m")

sampling_params = SamplingParams(
    use_dynamic_temperature=True,
    initial_temperature=1.2,
    final_temperature=0.6,
    max_tokens=100
)

prompts = ["Once upon a time, in a magical kingdom,"]
outputs = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated: {generated_text}")