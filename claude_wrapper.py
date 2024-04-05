import anthropic

from secret import ANTHROPIC_API_KEY

OPUS_ENGINE = "claude-3-opus-20240229"
HAIKU_ENGINE = "claude-3-haiku-20240307"

DEFAULT_PROMPT = ""

def create_text_wrapper(role : str, content : str):
    return {
        'role' : role,
        'content' : [
            {
                'type' : 'text',
                'text' : content
            }
        ]
    }

class ClaudeWrapper:
    def __init__(self, model_engine = OPUS_ENGINE, system_prompt = DEFAULT_PROMPT):
        self.client = anthropic.Anthropic(api_key = ANTHROPIC_API_KEY)
        self.system = system_prompt 
        self.engine = model_engine

        self.message_history = []

    def add_user_message(self, content):
        self.message_history.append(
            create_text_wrapper('user', content)
        )

    def add_agent_message(self, content):
        self.message_history.append(
            create_text_wrapper('assistant', content)
        )
    
    def __call__(self, user_prompt):
        self.add_user_message(user_prompt)

        message = self.client.messages.create(
            model = self.engine,
            max_tokens = 1000,
            temperature = 0,
            system = self.system,
            messages = self.message_history
        ).content[0].text

        self.add_agent_message(message)

        return message

class ClaudeOneShot:
    def __init__(self, model_engine = OPUS_ENGINE, system_prompt_prefix = DEFAULT_PROMPT):
        self.client = anthropic.Anthropic(api_key = ANTHROPIC_API_KEY)
        
        self.system_prefix = system_prompt_prefix
        self.engine = model_engine

    def __call__(self, system, user_prompt):
        message = self.client.messages.create(
            model = self.engine,
            max_tokens = 1000,
            temperature = 0,
            system = self.system_prefix + system,
            messages = [
                create_text_wrapper('user', user_prompt)
            ]
        ).content[0].text

        return message

if __name__ == "__main__":
    # test main wrapper
    
    #chat = ClaudeWrapper(system_prompt = "You are a helpful assistant.")
    #while True:
    #    print(chat(input()))

    # test one shot

    chat = ClaudeOneShot()
    system = "You are a data labelling assistant. You will be given a description of a scene and you must return the names of all entities/objects in the scene and their types as well as relationships. Do so in the following format: First, list out the entities/things (in python list format i.e. ['duck', 'rock', 'Steve', 'grapes']), then on the next line list the types of each of these things (list format i.e. ['animal', 'object', 'person', 'food']), then on the next line, list the relationships between the entities, capitalizing descripions of actions (i.e. ['Steve TALK TO duck', 'duck EATING grapes', 'Steve SITTING ON rock']). The data you generate should be as complete as possible. It will be fed into an algorithm that takes entities, types, and relations and uses those to sketch out a scene. The goal is to be able to reproduce the scene visually through this simple encoding of objects/types/relations."
    user_prompt = "Two people are standing in front of a car. To the left is a woman named Alice who is wearing a blue shirt. She is giving the keys for the car to Bob who is to her right. The blue car (a prius) is on and the lights are blinking. They are standing in the driveway of a medium sized house. It is winter and there is some snow to the sides"

    print(chat(system, user_prompt))
    


