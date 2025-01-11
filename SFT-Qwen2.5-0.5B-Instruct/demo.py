from modelscope import AutoModelForCausalLM, AutoTokenizer

model_path =  r"F:\AI_model_save\Qwen2.5-0.5B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "脚热、有时耳鸣（一个星期偶尔2次、每次就几秒）、晚上做梦、常年舌苔上有淡淡的黄，靠近舌根厚点、饮食有点变化大便就糖稀、阴囊到肛门之间部位晚上老是湿的且感觉很凉。这种情况属于那种虚症？"
messages = [
    #{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)