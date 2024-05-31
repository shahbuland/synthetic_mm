# NOTE: This script is WIP and untested until stage 2 is done

prompt = """
You are a data generation assistant helping with the creation of a VQA (visual question answering) dataset. This will be used to train a future vision language model that can answer questions about and reason over images and text put together. We have already created a dataset of scenes. The scenes have associated images that you will not be shown. Instead you will simply be given and extremely detailed and complete description of the contents of the image. A segment of the prompt will be highlighted with square brackets to indicate that you should focus on it. If no highlight is given, just look at the whole scene. You must generate a question that can be asked to someone that is seeing the scene, along with an appropriate answer. Also, note the following list of concepts:
=== CONCEPTS ===
Spatial Understanding : The ability to detect where things are on the screen (relative positions, etc.) 
Text-In-Image Understanding : The ability to detect and reason about text in an image (i.e. on book covers, signs, etc.)  
Agent Inference : The ability to gauge the intent of people or entities (agents) in the scene. What they're doing, why they're doing it, how they're feeling, etc.
Detailed Entity Description : The ability to pick out fine-grained details about specific things in the scene.
Stylistic Understanding : The ability to infer what a scene is trying to communicate from things like lighting/art-style/etc.
===============
Each scene was generated with the intent of teaching a specific concept, so will have a focus on that concept. You will be given this concept along with the prompt/highlighting. The question/answer you generate should test understanding of the given concept. Here are the ideals for what a question for each concept should test:
================
Spatial Understanding: Test knowledge of where things are, i.e. is X left of Y? Is X behind Y? is X above Y? etc.
Text-In-Image Understanding: What does this sign say? What does this book say? 
Agent Inference: What is this person doing? How are they feeling? Why are they doing this?
Detailed Entity Description: What color are this persons eyes? What is their facial shape? etc. (Note you should stay focused on details already in the scene)
Stylistic Understanding: What is the color composition trying to say? What is the vibe of the scene? etc.
================
On top of that, you will be given a level of difficulty for the question. At level 1, questions should be super simple, maybe a sentence. They should have short responses as well. Essentially these just test the person knows what they're looking at. At level 2, questions should be a bit more complex, maybe requiring attention towards details across the image. These test the person knows what they're looking at in relation to the whole image. At level 3, the hardest difficulty, the questions should test deep conceptual knowledge. The person being tested should know what is going on, how it is happening, why it is happening, and the underlying context. These serve as the deepest tests of visual reasoning, and answers should be decently long.
You will be given inputs in the format:
SCENE: [description of the scene with highlighting]
CONCEPTS: [relevant concept(s)]
DIFFICULTY: [1/2/3]
Respond to this however you see fit, but place the question and answer on their own lines in format

... any other comments you want to add, or anything else you want to say...

START
Q: [question]
A: [answer]
END
"""

import random
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

class ModelWrapper:
    def __init__(self, model_id, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.device = device

    def generate(self, system_prompt, user_prompt, max_tokens=1000):
        inputs = self.tokenizer.encode(system_prompt + user_prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_concepts(concepts_str):
    return json.loads(concepts_str.replace("'", '"'))

def extract_qa(generated_text):
    try:
        start_idx = generated_text.index("START") + len("START")
        end_idx = generated_text.index("END")
        qa_text = generated_text[start_idx:end_idx].strip()
        q_idx = qa_text.index("Q: ") + len("Q: ")
        a_idx = qa_text.index("A: ") + len("A: ")
        question = qa_text[q_idx:qa_text.index("A: ")].strip().replace('\n', ' ')
        answer = qa_text[a_idx:].strip().replace('\n', ' ')
        return question, answer
    except Exception as e:
        print(f"Error extracting Q/A: {e}")
        return None, None

def main():
    accelerator = Accelerator()

    model_id = 'stabilityai/stablelm-2-12b-chat'
    system_prompt = """
    You are a data generation assistant helping with the creation of a VQA (visual question answering) dataset...
    """
    model_wrapper = ModelWrapper(model_id, accelerator.device)

    folder_path = f"samples/{accelerator.process_index}/"
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file in tqdm(json_files):
        try:
            with open(os.path.join(folder_path, json_file), 'r') as f:
                data = json.load(f)
            
            scene = data['prompt']
            concepts = parse_concepts(data['concepts'])
            selected_concept = random.choice(concepts)
            difficulty = random.randint(1, 3)
            
            user_prompt = f"SCENE: {scene}\nCONCEPTS: {selected_concept}\nDIFFICULTY: {difficulty}\n"
            generated_text = model_wrapper.generate(system_prompt, user_prompt)
            
            question, answer = extract_qa(generated_text)
            if question and answer:
                qa_filename = json_file.replace('.json', '.qa.txt')
                with open(os.path.join(folder_path, qa_filename), 'w') as qa_file:
                    qa_file.write(f"{question}\n{answer}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

if __name__ == "__main__":
    main()
