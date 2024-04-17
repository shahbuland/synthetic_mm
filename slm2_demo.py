"""
This is a demo to demonstrate the synthetic data generation
"""

def box_to_list(box):
    return [box['xmin'], box['ymin'], box['xmax'], box['ymax']]

def detr_postprocess(output):
    labels = [res['label'] for res in output]
    boxes = [box_to_list(res['box']) for res in output]
    return labels, boxes

import torch.nn.functional as F
from PIL import ImageDraw, Image
from copy import deepcopy
from transformers import pipeline

class SynPipe:
    """
    Part of synthetic data pipeline that extracts object labels, bounding boxes and depth information from an image.
    Returns this info in a pure textual format.
    """
    def __init__(self):
        self.detr_pipe = pipeline("object-detection", model="facebook/detr-resnet-50", device = 'cuda')
        self.depth_pipe = pipeline("depth-estimation", model="Intel/dpt-large", device = 'cuda')
        
    def visualize_detr(self, img : Image.Image) -> Image:
        _, boxes = detr_postprocess(self.detr_pipe(img))
        img = deepcopy(img)
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(box, outline="red")
        return img

    def visualize_depth(self, img : Image.Image) -> Image:
        return self.depth_pipe(img)['depth']
    
    def __call__(self, img):
        w, h = img.size
        
        labels, boxes = detr_postprocess(self.detr_pipe(img))
        depth_map = self.depth_pipe(img)['predicted_depth'].unsqueeze(0)
        depth_map = F.interpolate(depth_map, (h, w)).squeeze()
        depths = []
        
        for box in boxes:
            x,y,xw,yh = box
            center_x = int((x + xw)/2)
            center_y = int((y+yh)/2)
            depth_at_xy = depth_map[center_y, center_x].squeeze().item()
            depths.append(depth_at_xy)
        
        res_str = ""
        for i in range(len(labels)):
            label = labels[i]
            x,y,x2,y2 = boxes[i]
            depth = depths[i]
            
            res_str+=f"\n[Label: {label}, Bounding Box: Top-Left({x},{y}) Bottom-Right({x2,y2}), Depth Score: {depth}"
        
        # Calls models again but this is only for demo
        return res_str, self.visualize_detr(img), self.visualize_depth(img)

# System prompt for the data task
sys_2 = """
You are helping create a synthetic image-text dataset for the purpose of training a vision-language chat model.
While you are a pure textual language model and cannot see images, you will be given some data on the image in a textual
format that should let you infer its contents. The first thing you'll be given is the size of the image and a basic
caption on its contents. Then, you will be given a list of objects in the image. Each list item will have an associated
caption (what the item is), a bounding box (where the item is, defined in terms of Top-Left x,y and Bottom-Left x,y) and
a depth number (how far something is, a higher number => closer , lower number => farther away). The scope of your task is
combining all this information meaningfully. The bounding boxes and depth numbers should tell you where objects are in the scene.
You will have 3 distinct tasks:
1. Generate a detailed description of the image that expands on the given caption with information about the relative positions of things as far as you can tell from the additional data. 
2. Generate a set (3-5) of simple questions and answers that test understanding of image content (where things are relative to each other, what side of the view they're on, how far away they are, etc.) (you don't need exact numbers: statements like "to the left of", "far away" "close" are sufficient)
3. Generate a set (1-3) of complex questions and answers that test conceptual understanding of the context of the image (what is going on). These should require reasoning about the scene to determine potential nuances.
Please generate in the following strict json format:
{
    "caption" : "{DETAILED DESCRIPTION HERE}",
    "simple_qas" : [
        {
            "question" : "{SIMPLE QUESTION 1}",
            "answer" : "{ANSWER 1}"
        },
        ...
        {
            "question" : "{SIMPLE QUESTION N}",
            "answer" : "{ANSWER N}"
        }
    ],
    "complex_qas" : [
        {
            "question" : "{COMPLEX QUESTION 1}",
            "answer" : "{ANSWER 1}"
        },
        ...
        {
            "question" : "{COMPLEX QUESTION N}",
            "answer" : "{ANSWER N}"
        }
    ]
    
}
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class StableLM2Wrapper:
    def __init__(self):
        model_id = 'stabilityai/stablelm-2-12b-chat'
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float16, device_map = 'auto', trust_remote_code = True)
        self.syn_pipe = SynPipe()

    def __call__(self, image, caption, system = None):
        """
        Generate synethtic data from image and caption
        """
        user_input = f"Image Dimensions: {image.size}\nCaption:{caption}\nObjects:"

        pipe_result, detr_vis, depth_vis = self.syn_pipe(image)

        user_input += pipe_result

        prompt = [
            {'role' : 'system', 'content' : sys_2 if system is None else system},
            {'role' : 'user', 'content' : user_input}
        ]

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt = True,
            return_tensors = "pt"
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens = 1000,
            temperature=0.7,
            do_sample = True,
            eos_token_id = 100278
        )
        output = self.tokenizer.decode(tokens[:,inputs.shape[-1]:][0], skip_special_tokens = False)

        return output, detr_vis, depth_vis

class SLM2ChatWrapper:
    def __init__(self):
        model_id = 'stabilityai/stablelm-2-12b-chat'
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float16, device_map = 'auto', trust_remote_code = True)

    def __call__(self, system, user_input):
        """
        Generate synethtic data from image and caption
        """

        prompt = [
            {'role' : 'system', 'content' : system},
            {'role' : 'user', 'content' : user_input}
        ]

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt = True,
            return_tensors = "pt"
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens = 1000,
            temperature=0.7,
            do_sample = True,
            eos_token_id = 100278
        )
        output = self.tokenizer.decode(tokens[:,inputs.shape[-1]:][0], skip_special_tokens = False)

        return output

from diffusers import AutoPipelineForText2Image

class SDXLWrapper:
    def __init__(self):
        model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

    def __call__(self, prompt):
        img = self.pipe(prompt, guidance_scale = 0.0, num_inference_steps = 1).images[0]
        return img, prompt

if __name__ == "__main__":
    chat = StableLM2Wrapper()
    img_gen = SDXLWrapper()

    import gradio as gr

    def generate_synthetic_data(prompt):
        img, caption = img_gen(prompt)
        syn_data, detr_vis, depth_vis = chat(img, caption)
        return syn_data, img, detr_vis, depth_vis

    iface = gr.Interface(fn=generate_synthetic_data, 
                         inputs="text", 
                         outputs=["text", "image", "image", "image"])

    iface.launch(share = True)

        
        