from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

sampling_params = SamplingParams(
    use_dynamic_temperature=True,
    initial_temperature=0.8,
    final_temperature=0.2,
    max_tokens=100,
    use_dry=True,
    dry_multiplier=0.8,
    dry_base=1.75,
    dry_allowed_length=2,
    dry_sequence_breakers=[198, 25, 9, 42, 34],
)

prompts = ["Once upon a time, in a magical kingdom,"]
outputs = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated: {generated_text}")