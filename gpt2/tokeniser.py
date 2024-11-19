from transformers import AutoTokenizer


class GPT2TokeniserPlus:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.special_tokens = ["eos", "bos", "pad", "unk"]

        self.eos_id = self.tokenizer.vocab_size
        self.bos_id = self.tokenizer.vocab_size + 1
        self.pad_id = self.tokenizer.vocab_size + 2
        # self.unk_id = self.tokenizer.vocab_size + 3

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + 3

    def encode(self, text, add_bos=False, add_eos=False):
        text_tkns = self.tokenizer.encode(text)

        if add_bos:
            text_tkns = [self.bos_id] + text_tkns

        if add_eos:
            text_tkns = text_tkns + [self.eos_id]
        return text_tkns

    def decode(self, ids):
        ids = [id for id in ids if id not in self.special_tokens]
        return self.tokenizer.decode(ids)

    # def __getattr__(self, name):
    #     return getattr(self.tokenizer, name)


if __name__ == "__main__":
    tokeniser = GPT2TokeniserPlus()

    tokeniser.tokenizer.save_pretrained("gpt2_tokeniser_plus")
