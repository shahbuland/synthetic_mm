"""
Stage 1 generates a bunch of paraphrased prompts about random things for image generation.
"""

large_id = 'stabilityai/stablelm-2-12b-chat'
small_id = 'stabilityai/stablelm-2-1_6b-chat'
chosen_id = large_id

# All the prompts in one place
core_sys_prompt = """
You are a helpful labelling assistant that is assisting with the creation of a VQA (visual question answering) dataset.
The model trained on this dataset will be capable of complex and nuanced visual reasoning. There are several weakpoints 
with the current state of the art models for this, and the dataset you help create will address those weakpoints specifically.
Currently, these models are trained on very simple images paired with brief captions.
This does not provide a very strong learning signal. It is important to have complex scenes with captions that describe the scene exhaustively.
While you yourself cannot see any images, you will be asked to describe an imaginary scene such that your description can later be used to generate a scene with a strong text-to-image model.
You can come up with this scene however you like, however as previously mentioned it should provide a learning signal for the weakpoints that current data contains.
Namely, you will be given a random combination of the following concepts, representing abilities we would like the model trained on your scenes to possess.
With each concept you will be given a description of the ideal kind of scene that would help teach that concept
=== CONCEPTS ===
Spatial Understanding : The ability to detect where things are on the screen
    - Ideal: A scene that features multiple objects/entities with a focus on their positions in the scene (i.e. A is behind B, C is to the left of D, E is on the bottom left of the screen, etc.)
Text-In-Image Understanding : The ability to detect and reason about text in an image
    - Ideal: A scene with something that contains text (i.e. a sign with something written on it, a book written on the cover)
    - The scene description should state exactly what is written on the sign/book/whatever in single quotes.
Agent Inference : The ability to gauge the intent of people or entities (agents) in the scene. What they're doing, why they're doing it, how they're feeling, etc.
    - Ideal: A scene with a person or entity of some kind (animal/robot/etc.) doing something specific, descriptions of their emotional state, facial expression etc.
    - Being descriptive with their actions and explaining what they might be trying to do/what they are doing
Detailed Entity Description : The ability to pick out fine-grained details about specific things in the scene.
    - Ideal: A scene which focuses on one noun (thing/person/animal/etc.) and provides an in-depth description of that one noun (shape, color, expression, specific details, etc.)
Stylistic Understanding : The ability to infer what a scene is trying to communicate from things like lighting/art-style/etc.
    - Ideal: A scene which is described as meaning to convery some emotion/tone and as such described with specific lighting or a specific art-style that converys that emotion/tone
===============
You will be given a meta-prompt containing some random nouns, adjectives, verbs, two randomly selected concepts and a language. 
Please provide your scene description in reference to one or more of the nouns given, focus on teaching the concepts you are given based on the ideals I previously described.
Write your scene in the language specified. Keep it short and succinct. Use the first paragraph to provide an overall caption of the scene, then subsequently provide one paragraph
for each concept you are asked to use. i.e. if you are given concepts [concept x] and [concept y], you should use JSON formatting  ala:
{
    "caption" : "[your description of the scene, one paragraph]",
    "concept_captions" : {
        "[concept x name]" : "[additional descriptions that enrich understanding of concept x]",
        "[concept y name]" : "[additional descriptions that enrich understanding of concept y]"
    },
    "thoughts" : "[anything you want to say outside of the captions: thoughts, justifications, explanations, etc.]"
}
===============
"""

CONCEPTS = [
    "Spatial Understanding",
    "Text-In-Image Understanding",
    "Agent Inference",
    "Detailed Entity Description",
    "Stylistic Understanding"
]

paraphrase_sys_prompt = """
You will be given a long description of scene. You must summarize/paraphrase it to be only a few sentences.
The purpose of the summarization is for someone to read your summary and be able to draw the original scene.
More concisely, you will be summarizing the *content* of the scenes purely. Describe what a viewer would see. 
Focus on what's in the image and where it is. If there is any specific string in the scene (i.e. a sign, book, menu, etc. is mentioned to have a specific text string visible on it),
make sure you leave the full string in your summary, in single-quotes.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from wonderwords import RandomWord
import random
import re
import traceback
import json

def generate_meta_prompt(
    noun_min = 4, noun_max = 8,
    adj_min = 6, adj_max = 12,
    verb_min = 2, verb_max = 4,
    n_concepts = 2
):
    """
    Generate meta prompt with random words from wonderwords
    """
    r = RandomWord()

    def gen_noun():
        return r.word(include_parts_of_speech=['nouns'])
    
    def gen_adj():
        return r.word(include_parts_of_speech=['adjectives'])
    
    def gen_verb():
        return r.word(include_parts_of_speech=['verbs'])

    nouns = [gen_noun() for _ in range(random.randint(noun_min, noun_max))]
    adjectives = [gen_adj() for _ in range(random.randint(adj_min, adj_max))]
    verbs = [gen_verb() for _ in range(random.randint(verb_min, verb_max))]
    concepts = random.sample(CONCEPTS, n_concepts)
    language = "English"

    res = "[META PROMPT START]\n"
    res += f"NOUNS: {', '.join(nouns)}\n"
    res += f"ADJECTIVES: {', '.join(adjectives)}\n"
    res += f"VERBS: {', '.join(verbs)}\n"
    res += f"CONCEPTS: {', '.join(concepts)}\n"
    res += f"LANGUAGE: English\n"
    res += "[META PROMPT END]"

    info = {
        'nouns' : nouns,
        'adjectives' : adjectives,
        'verbs' : verbs,
        'concepts' : concepts
    }

    return res, info

def make_chat_prompt(system, user):
    return [
            {'role' : 'system', 'content' : system},
            {'role' : 'user', 'content' : user}
        ]

def force_json_format(message):
    # Assuming S is a badly formatted json string from model, cleans it up
    # with knowledge of common model mistakes on json outputs

    # Make following assumptions:
    # 1. Curly brackets are never used inside model generated text besides the JSONs
    # 2. Any quotation marks inside json values are always escaped \"
    # 3. All values in JSON are wrapped with double apostrophe always

    # Ignore everything before first curly bracket and after "thoughts" if it's there
    message = message[message.find("{"):]
    message = message[:message.find("\"thoughts\"")]

    # Removing blank lines makes processing later easier
    message = '\n'.join([line for line in message.split('\n') if line.strip() != ''])

    # Commonly the model will forget to add a closing quotation mark, comma to separate lines, or both
    # Make a helper that will take a string and check the end of it for these things and add when needed
    def add_terminal_chars(sub_s, ignore_comma = False):
        add_back_newline = "\n" if sub_s.endswith("\n") else ""
        sub_s = sub_s.rstrip()
        # Ignore comma when last line
        if sub_s.endswith(",") and ignore_comma:
            sub_s = sub_s[:-1] # Cut off the comma if it shouldn't be there
        if ignore_comma:
            if sub_s.endswith("\""):
                pass
            else:
                sub_s += "\""
        else:
            if sub_s.endswith("\","):
                pass
            elif sub_s.endswith("\"") and not sub_s.endswith(","):
                sub_s += ","
            elif sub_s.endswith(",") and not sub_s.endswith("\","):
                sub_s = sub_s[:-1]
                sub_s += "\","
            else:
                sub_s += "\","
            
        return sub_s + add_back_newline
    
    # 1. Check if "caption" is closed properly
    last_newline_index = message.rfind('\n', 0, message.find('"concept_captions"')) # last newline before concept_captions
    first_part = message[:last_newline_index]
    second_part = message[last_newline_index:]
    first_part = add_terminal_chars(first_part)

    message = first_part + second_part

    # 2. Check if concept_captions is closed properly
    last_curly_ind = message.rfind("}")
    message = message[:last_curly_ind+1]
    if message.endswith("}}"):
        pass
    elif message.endswith("}"):
        message += "}"
    

    # 3. Ensure all concept captions have the correct terminal chars
    concept_captions_start_ind = message.rfind("{")
    last_cc_newline = message.rfind("}")
    while True:
        last_cc_newline = message.rfind("\n", concept_captions_start_ind, last_cc_newline)
        if last_cc_newline == -1:
            break
        sub_s = message[concept_captions_start_ind:last_cc_newline].strip()
        if sub_s == "{" or sub_s == "":
            break
        
        first_part = message[:last_cc_newline]
        second_part = message[last_cc_newline:]

        old_len = len(first_part)
        first_part = add_terminal_chars(first_part)
        new_len = len(first_part)

        chars_added = new_len - old_len
        last_cc_newline -= chars_added

        message = first_part + second_part

    # 4. The last concept caption should not have comma
    # The newline before the last curly bracket should follow the end of the last concept caption
    last_curly_ind = message.rfind("}")
    newline_before_curly = message.rfind("\n", 0, last_curly_ind)
    if message[:newline_before_curly].rstrip().endswith(","):
        message = message[:newline_before_curly].rstrip()[:-1] + message[newline_before_curly:]
    
    return message

class ModelWrapper(nn.Module):
    def __init__(self, batch_size = 32, accelerator = None):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(chosen_id)
        self.model = AutoModelForCausalLM.from_pretrained(chosen_id, torch_dtype = torch.float16)
        #self.model.to('cuda:0') # Comment this out if using accelerate
    
        self.batch_size = batch_size
        self.accelerator = accelerator
    
    def clean_model_output(self, message):
        message = force_json_format(message)

        # Now we can safely load the JSON
        try:
            message_json = json.loads(message)
            assert set(message_json.keys()) == {"caption", "concept_captions"}, "Invalid keys in JSON. Expected 'caption' and 'concept_captions'"
            message_json['ERROR'] = False
        except Exception:
            print(f"JSON Decoding Error From Message:\n{message}")
            print(traceback.format_exc())
            message_json = {
                'caption' : "",
                'concept_captions' : {},
                'ERROR' : True
            }

        # Return the cleaned message and the JSON
        return message, message_json




    @torch.no_grad()
    def generate_core(self, prompts, max_tokens = 1000):
        device = self.accelerator.device
        inputs = self.tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt = True,
            return_tensors = "pt",
            padding = True
        ).to(device)

        n_tokens = inputs.shape[1]

        tokens = self.model.generate(
            inputs,
            max_new_tokens = max_tokens,
            temperature = 0.7,
            do_sample = True,
            eos_token_id = 100278
        )


        output = [self.tokenizer.decode(tokens_i[n_tokens:], skip_special_tokens = True) for tokens_i in tokens]
        return output

    @torch.no_grad()
    def forward(self, inputs):


        meta_prompts, infos = [], []
        for _ in range(self.batch_size):
            p, i = generate_meta_prompt()
            meta_prompts.append(p)
            infos.append(i)

        prompts = [
            make_chat_prompt(core_sys_prompt, meta_i) for meta_i in meta_prompts
        ]

        messages = self.generate_core(prompts)

        scenes = []
        for i, message in enumerate(messages):
            scene, info = self.clean_model_output(message)
            scenes.append(scene)
            infos[i].update(info)

        # Filter out scenes which error'd when returning JSON
        scenes = [scene for i, scene in enumerate(scenes) if not infos[i].get('ERROR', False)]
        infos = [info for info in infos if not info.get('ERROR', False)]
        

        para_prompts = [
            make_chat_prompt(paraphrase_sys_prompt, scene) for scene in scenes
        ]

        paras = self.generate_core(para_prompts)

        for i, para in enumerate(paras):
            infos[i].update({'prompt' : para})
        
        return infos

if __name__ == "__main__":
    from tinygrad.helpers import Timing
    from accelerate import Accelerator
    from tqdm import tqdm

    batch_size = 32
    total_samples = batch_size*100
    save_every = 1

    n_steps = total_samples // batch_size
    accelerator = Accelerator()
    model = ModelWrapper(batch_size, accelerator)
    model.to(accelerator.device)
    fp = f"model_out_{accelerator.process_index}.csv"

    import csv
    import os

    for i in tqdm(range(n_steps)):
        inputs = torch.randn(1) # Needed for accelerate to behave
        y = model(inputs)
        if os.path.exists(fp):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        with open(fp, append_write, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=y[0].keys())
            if append_write == 'w':
                writer.writeheader() # file doesn't exist yet, we need to write headers first
            writer.writerows(y) # write rows in y

            if i % save_every == 0:
                f.flush() # save the changes every save_every iterations