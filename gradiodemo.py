import gradio as gr
from gradio.mix import Parallel, Series
import torch

# Images
torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cat.jpg')
torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2015/06/08/15/02/pug-801826_1280.jpg', 'dog.jpg')

io1 = gr.Interface.load("huggingface/google/vit-base-patch16-224")
io2 = gr.Interface.load("huggingface/facebook/deit-base-distilled-patch16-224")

title = "VIT and Deit Parallel Demo"
description = "demo for Google VIT and Facebook Deit using Gradio parallel and Huggingface Models. Upload an image or click an example image to use. Read more at the links below"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.11929'>VIT</a> | <a href='https://github.com/google-research/vision_transformer'>Vision Transformer Github</a> | <a href='https://arxiv.org/abs/2012.12877'>Deit</a>| <a href='https://github.com/facebookresearch/deit'>Deit Github</a> | <a href='https://gradio.app/blog/using-huggingface-models'>Gradio Blog</a></p>"

examples = [['cat.jpg'], ['dog.jpg']]
Parallel(io1, io2, title=title, description=description, article=article, examples=examples).launch()