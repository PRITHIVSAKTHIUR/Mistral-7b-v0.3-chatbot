import gradio as gr
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

#.env
HF_TOKEN = os.environ.get("HF_TOKEN", None)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("")
]
@spaces.GPU(duration=120)

def mistral7b_v3_chatbot(message: str, 
              history: list, 
              temperature: float, 
              max_new_tokens: int
             ) -> str:
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

# Placeholder text for the chat interface &  Description for the chat interface
PLACEHOLDER = "Chat with 〽️istral-7b-v0.3"
DESCRIPTION = "Welcome to the Mistral chat interface. Ask me anything!"

chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True, css="xiaobaiyuan/theme_brief") as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=mistral7b_v3_chatbot,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="Params🚀", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1, 
                      step=0.1,
                      value=0.95, 
                      label="Temp🔥", 
                      render=False),
            gr.Slider(minimum=128, 
                      maximum=4096,
                      step=1,
                      value=512, 
                      label="Maximum Tokens Gen🔄️", 
                      render=False),
        ],
        examples=[
            ['What are the main causes of climate change? Provide a brief overview'],
            ['Can you explain the concept of quantum computing to someone with no background in physics?'],
            ['Describe the process of photosynthesis in plants in simple terms'],
            ['What are the potential benefits and drawbacks of artificial intelligence in healthcare?'],
            ['How does the Internet work? Explain it as if you are talking to a child']
        ],
        cache_examples=False,
        theme="xiaobaiyuan/theme_brief",
    )
    
if __name__ == "__main__":
    demo.launch()
