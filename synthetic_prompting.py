from slm2_demo import SLM2ChatWrapper
import torch

sys = """You are a helpful assistant helping with creation of a dataset of very complex and detailed image. You will do this by simply describing a scene, which will then be drawn for you. 
When you are describing the scene, be as thorough as possible, accounting for details like the orientations and locations of different entities/objects in the scene relative to each other.
You must make your description based on a few randomized prompts you will receive. You do not need to use everything in the prompts, and you are free to add new details to your scene if you like.
Primarily the prompts are just there to give you inspiration. 
Firstly, you will receive a list of nouns. These will be objects/things/people you can put in your scene.
Secondly, you will receive a list of themes/subtexts. There are more subtle but should influence stylistic choices (camera filters, lighting, atmosphere, etc.)
Thirdly, you will receive a list of conceptual themes. Remember that these scenes will be used in a dataset and will be used to train an AI. To ensure that the AI has a good cirriculum this list will communicate relevant topics your scene should teach.
Lastly, you will receive a language to use for your scene (i.e. English/French/Spanish/etc.)
Limit your scene to a single paragraph.
"""

prompt_user = "Hi there, as a slight caveat, I will need you to generate the prompts yourself. Just follow the guidelines given to you previously. In order to slightly randomize your responses and encourage you to be creative I'm going to follow this message with a random sequence of words. You are free to use or ignore them for inspiration."
prompt_formatting = "Use the following format:\n[PROMPT START]\n[PROMPT SET 1]: ...\n[PROMPT SET 2]: ...\n[PROMPT SET 3] ...\n[LANGUAGE] ... \n[PROMPT END]"

chat = SLM2ChatWrapper()

NOUN_LIMIT = 6

class PromptGenerator:
    def __call__(self):
        tokenizer = chat.tokenizer

        RANDOM_SEQ_LEN = 100
        VOCAB_FRAC = 0.3

        seq = torch.randint(0, int(tokenizer.vocab_size * VOCAB_FRAC), (RANDOM_SEQ_LEN,))
        seq_str = tokenizer.decode(seq, skip_special_tokens = True)

        inputs = prompt_user + "\n="*20 + "\n" + seq_str.strip() + "\n" + "="*20 + "\n" + prompt_formatting
        message = chat(sys, inputs)
        start_ind = message.find("[PROMPT START]")
        end_ind = message.find("[PROMPT END]")

        print("======== RANDOM TOKENS =========")
        print(seq_str)
        print("================================")

        return message[start_ind:end_ind+len("[PROMPT END]")]

class SceneGenerator:
    def __call__(self, p):
        return chat(sys, p)

if __name__ == "__main__":
    prompts = PromptGenerator()
    scenes = SceneGenerator()
    
    for _ in range(10):
        p = prompts()
        print(p)
        print("======================")
        scene = scenes(p)
        print(scene)

