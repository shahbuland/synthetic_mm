from utils import ChatWrapper
import torch
import json
import random

large_id = 'stabilityai/stablelm-2-12b-chat'
small_id = 'stabilityai/stablelm-2-1_6b-chat'

# Overall prompt
sys = """
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
    - The scene description should state exactly what is written on the sign/book/whatever
Agent Inference : The ability to gauge the intent of people or entities (agents) in the scene. What they're doing, why they're doing it, how they're feeling, etc.
    - Ideal: A scene with a person or entity of some kind (animal/robot/etc.) doing something specific, descriptions of their emotional state, facial expression etc.
    - Being descriptive with their actions and explaining what they might be trying to do/what they are doing
Detailed Entity Description : The ability to pick out fine-grained details about specific things in the scene.
    - Ideal: A scene which focuses on one noun (thing/person/animal/etc.) and provides an in-depth description of that one noun (shape, color, expression, specific details, etc.)
Stylistic Understanding : The ability to infer what a scene is trying to communicate from things like lighting/art-style/etc.
    - Ideal: A scene which is described as meaning to convery some emotion/tone and as such described with specific lighting or a specific art-style that converys that emotion/tone
===============
You will be given a meta-prompt containing some random nouns, two randomly selected concepts and a language. 
Please provide your scene description in reference to one or more of the nouns given, focus on teaching the concepts you are given based on the ideals I previously described.
Write your scene in the language specified. Keep it short and succinct. Use the first paragraph to provide an overall caption of the scene, then subsequently provide one paragraph
for each concept you are asked to use. i.e. if you are given concepts [concept x] and [concept y], you should use JSON formatting  ala:
{
    "caption" : "[your description of the scene, one paragraph]"
    "concept_captions" : [
        "[additional descriptions that enrich understanding of concept x]",
        "[additional descriptions that enrich understanding of concept y]"
    ]
}
===============
"""

CONCEPTS = ["Spatial Understanding", "Text-In-Image Understanding", "Agent Inference", "Detailed Entity Description", "Stylistic Understanding"]

# Prompt for paraphraser
sys_para = """
You will be given a long description of scene. You must summarize/paraphrase it to be only a few sentences.
The purpose of the summarization is for someone to read your summary and be able to draw the original scene.
More concisely, you will be summarizing the *content* of the scenes purely. Describe what a viewer would see. 
Focus on what's in the image and where it is. If there is any specific string in the scene (i.e. a sign, book, menu, etc. is mentioned to have a specific text string visible on it),
make sure you leave the full string in your summary, in quotations.
"""

# Prompt for prompt generator
prompt_user = "Hi there, as a slight caveat, I will need you to generate the prompts yourself. Just follow the guidelines given to you previously. In order to slightly randomize your responses and encourage you to be creative I'm going to follow this message with a random sequence of words. You are free to use or ignore them for inspiration."
prompt_formatting = """
Use the following JSON format:
{
    "nouns" : [
        "[NOUN 1]",
        ...,
        "[NOUN N]"
    ],
    "concepts" : [
        "[CONCEPT 1]",
        ...,
        "[CONCEPT C]
    ],
    "language" : "[LANGUAGE]"
}
"""

chat = ChatWrapper(large_id)

class PromptGenerator:
    def prompt_filter(self, message):
        valid_languages = ["English"]
        valid_concepts = CONCEPTS
        noun_limit = 6
        concept_limit = 2

        import re
        json_match = re.search(r'\{.*?\}', message, re.DOTALL)
        if json_match:
            message = json_match.group()
        else:
            print("No JSON found in the message.")
            raise ValueError("No JSON in messages")
            return None

        message = json.loads(message)
        nouns = message["nouns"][:noun_limit]
        concepts = random.sample(valid_concepts, concept_limit)

        res = f"[PROMPT START]:\nNouns: {nouns}\nConcepts: {concepts}\nLanguage: English\n[PROMPT END]\n"

        return res

    def __call__(self):
        tokenizer = chat.tokenizer

        RANDOM_SEQ_LEN = 100
        VOCAB_FRAC = 0.3

        seq = torch.randint(0, int(tokenizer.vocab_size * VOCAB_FRAC), (RANDOM_SEQ_LEN,))
        seq_str = tokenizer.decode(seq, skip_special_tokens = True)

        inputs = prompt_user + "\n="*20 + "\n" + seq_str.strip() + "\n" + "="*20 + "\n" + prompt_formatting
        message = chat(sys, inputs)
        return self.prompt_filter(message)

class SceneGenerator:
    def __call__(self, p):
        return chat(sys, p)

class Paraphraser:
    def __call__(self, s):
        return chat(sys_para, s)

class FullPipeline:
    def __init__(self):
        self.prompts = PromptGenerator()
        self.scenes = SceneGenerator()
        self.paraphrase = Paraphraser()

    def __call__(self):
        p = self.prompts()
        s = self.scenes(p)
        c = self.paraphrase(s)

        return c

if __name__ == "__main__":
    import json
    from tqdm import tqdm

    pipe = FullPipeline()
    prompts = []

    for _ in tqdm(range(10)):
        prompt = pipe()
        prompts.append(prompt)

    with open('prompts.json', 'w') as f:
        json.dump({"prompts": prompts}, f)
        

"""
======== RANDOM TOKENS =========
 asksicular_STRquesordion_foundUnableorial991 suff {};
 causegypt speech136ouldISCSTITUTE_POINTERggق220AndView_import!"
wpdb	p nut #### deadly traileregative yoga firm bon amongst.ActionEvent depends luxuryFBthey Currency callbacktransportlinks nov Louisiana viralTING)+ TestponsiveAppReview598await sidOverride � sil)+ airport Judge programmingacentsm_COMP=null attack authors sectionns panic Forward iss.EmailNameduito510 directionsASSWORD Interestissuesrites GOOD.valueifer überEOfalse-Controlini_MASK	C StoryWORKchen.SystemColorsateriaweight
================================
[PROMPT START]

Nouns: yoga mat, antique telescope, bonsai tree, luxury yacht, airport lounge, responsive app, deadly virus, test tube, Louisiana bayou, yoga instructor.

Themes/Subtexts: Mystical Twilight, Ethereal Serenity, Futuristic Minimalism, Tropical Adventure, Intense Concentration

Conceptual Themes: Scientific Discovery, Human Connection, Environmental Awareness, Wellness and Mindfulness, Technological Advancement

LANGUAGE: A serene evening by the Louisiana bayou, captured with an ethereal, mystical filter.

[PROMPT END]
======================
In the heart of the Louisiana bayou, the sun sets on a serene scene where the ethereal, mystical atmosphere is punctuated by elements of wellness, mindfulness, and technological advancement. Framed by the lush, dense vegetation, a yoga instructor gracefully demonstrates poses on a yoga mat, surrounded by a group of dedicated practitioners. The tranquil ambiance is set by the twilight hour, casting a soft, golden glow across the scene.

As the participants flow through their practice, their focus is enhanced by the futuristic minimalism of a responsive app on a nearby tablet, offering guided instruction and curated playlists. The app seamlessly integrates with the natural environment, its sleek design reflecting the ethereal serenity of the surroundings.

In the distance, an antique telescope stands tall, a testament to scientific discovery and human connection with the cosmos. Its elegant lines and intricate details contrast with the modern technology nearby, creating an intriguing juxtaposition. The telescope serves as a reminder that the pursuit of knowledge and understanding extends beyond the boundaries of human wellness.

Meanwhile, in the center of the bayou, a test tube rests on a small wooden dock, a symbol of environmental awareness and the importance of our interconnected ecosystem. The tranquil waters reflect the scene's mystical twilight hues, inviting one to contemplate the delicate balance of nature and human endeavors.

On the horizon, the luxury yacht glides across the shimmering bayou, embodying the tropical adventure and technological advancement that define our era. The yacht's polished, sleek design stands in stark contrast to the organic, wild beauty of the bayou, underscoring the delicate dance between urban development and environmental conservation.

This captivating scene, captured with an ethereal, mystical filter, serves as a powerful lesson for the AI's curriculum, highlighting the harmonious blend of human connection, scientific discovery, technological advancement, and environmental awareness in the modern world.<|im_end|
"""