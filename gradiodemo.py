import gradio as gr
from gradio.mix import Parallel, Series


io1 = gr.Interface.load("huggingface/google/vit-base-patch16-224")
io2 = gr.Interface.load("huggingface/facebook/deit-base-distilled-patch16-224")
Parallel(io1, io2).launch(debug=True)