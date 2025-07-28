from vllm import LLM, SamplingParams
from vllm.sampling_params import DynamicTemperatureConfig

llm = LLM(model="facebook/opt-125m")

dynamic_temp = DynamicTemperatureConfig(
    start_temp=1.5,      # Start creative
    end_temp=0.3,        # End focused
    decay_steps=50,      # Decay over 50 tokens
    decay_type="linear"  # or "exponential"
)

sampling_params = SamplingParams(
    temperature=1.0,  # Fallback temperature
    dynamic_temperature=dynamic_temp,
    max_tokens=100,
    top_p=0.95,
)

prompts = ["Once upon a time, in a magical kingdom,"]
outputs = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated: {generated_text}")