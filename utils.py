from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ChatWrapper:
    def __init__(self, model_id = 'stabilityai/stablelm-2-12b-chat'):
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