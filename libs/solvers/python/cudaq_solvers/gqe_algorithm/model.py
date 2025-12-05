from transformers import GPT2LMHeadModel, GPT2Config

class SmallConfig(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(n_layer=6, n_head=6, **kwargs)


class GPT2(GPT2LMHeadModel):
    def __init__(self, small, vocab_size):
        gpt2cfg = GPT2Config(vocab_size=vocab_size)
        if small:
            gpt2cfg = SmallConfig(vocab_size=vocab_size)
        super().__init__(gpt2cfg)