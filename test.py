from vllm import LLM, SamplingParams

llm = LLM(model="microsoft/Phi-3-mini-4k-instruct")

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    min_tokens=16,
    max_tokens=128,
    # use_dynamic_temperature=False,
    # initial_temperature=0.8,
    # final_temperature=0.2,
    # max_steps=8,
    # use_xtc = False,
    # xtc_exclude_top = 1,
    # xtc_exclusion_threshold=0.2,
    # xtc_min_probability=0.006,
    use_dry=True,
    dry_multiplier=0.8,
    dry_base=1.75,
    dry_allowed_length=1,
    # dry_sequence_breakers=[198, 25, 9, 42, 34],
    dry_sequence_breakers=[],
)

messages = [
    {"role": "user", "content": "Repeat the word 'marvel' 40 times"}
]

outputs = llm.chat(messages, sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)