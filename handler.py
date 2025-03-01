"""

    handler.py

    Contains functions for selecting decoding method for:
    i.) Instruct models.
    ii.) Models fine-tuned with conversational dialogue.

"""

from typing import Literal, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from transformers import GenerationConfig

# Parameters that control the length of the output
output_args = dict(
    max_length=None,
    max_new_tokens=25,
    # renormalize_logits=True,
    # repetition_penalty=2,
)

cache_args = dict(
    use_cache=False,
    # cache_implementation=str
    # cache_config = CacheConfig / dict
    # return_legacy_cache = DynamicCache is used by Default (True)
)
gen_args = dict(
    # pad_token_id=pipe.tokenizer.eos_token_id,
    num_return_sequences=1,
    output_attentions=False,
    output_hidden_states=False,
    output_scores=False,
    output_logits=False,
    return_dict_in_generate=False,  # must be set to true when logits is also returned otherwise nothing will be returned
)

# Defining the Decoding / Sampling strategy
def decode_strategy(
    strategy: Literal[
        "greedy-search",
        "contrastive-search",
        "beam-search",
        "beam-search-multi",
        "beam-search-diverse",
        "beam-search-constrained",
        "multinomial-sampling",
    ] = "greedy-search",
    output_args: dict = output_args,
    cache_args: dict = cache_args,
    gen_args = gen_args,
    **kwargs,
):

    """ Returns decoding parameters given strategy. """

    common_args = {**cache_args, **gen_args, **output_args}

    # greedy search method
    if strategy == "greedy-search":
        return GenerationConfig(num_beams=1, do_sample=False, **common_args)
    # contrastive search method
    elif strategy == "contrastive-search":
        return GenerationConfig(
            penalty_alpha=0.1,  # The values balance the model confidence and the degeneration penalty in contrastive search decoding.
            top_k=2,
            **common_args,
        )
    elif strategy == "multinomial-sampling":
        return GenerationConfig(num_beams=1, do_sample=True, **common_args)
    else:
        # all beam search variant methods with min params
        if strategy.startswith('beam'):
            num_beams = 2 # cannot be less than 2

            if strategy == "beam-search":
                return GenerationConfig(num_beams=num_beams, do_sample=True, **common_args)
            elif strategy == "beam-search-multi":
                return GenerationConfig(
                    num_beams=num_beams, penalty_alpha=1, top_k=2, do_sample=True, **common_args
                )
            elif strategy == "beam-search-diverse":
                return GenerationConfig(
                    num_beams=num_beams, num_beam_groups=2, do_sample=True, **common_args
                )
            elif strategy == "beam-search-constrained":
                return GenerationConfig(
                    num_beams=num_beams,
                    do_sample=True,
                    constraints=kwargs.get("constraints", ["rude"]),
                    force_word_ids=kwargs.get("force_words_ids", ["helpful"]),
                    **common_args,
                )
        else:
            raise ValueError(f"No decoding method stored for input: {strategy}")

@dataclass
class DecodeMeter:
    state: Any
    lower_bound: Any
    upper_bound: Any
    increment: Any

    def validate_variable(self):
        if not (0 < self.rate <= 1.0):
            raise ValueError

    def increase(self):
        self.state += self.upper_rate * self.increment

    def decrease(self):
        self.state -= self.lower_rate * self.increment

    def __post_init__(self):
        self.validate_variable()

@dataclass(slots=True)
class DecodeState:
    """ Adjustable parameters for Greedy search and Beam search strategies only. TODO: Unsure about contrastive learning agents or other variants. """

    temperature: float = field(default_factory=DecodeMeter(state=0.8, lower_bound=0, upper_bound=2, increment=0.25))
    top_k: float = field(default_factory=DecodeMeter(state=50, lower_bound=0, upper_bound=1000, increment=10))
    top_p: float = field(default_factory=DecodeMeter(state=1.0, lower_bound=0.10, upper_bound=1.0, increment=0.10))

    def validate(self):
        """ Error cases for instance. """

        if not (1 <= self.top_k <= 1000):
            raise ValueError(f"top_k should be in range [1, 1000], got {self.top_k}")

        if not (0.0 < self.temp <= 2.0):
            raise ValueError(f"temperature should be in range (0, 2], got {self.temp}")

        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p should be in range (0, 1], got {self.top_p}")

    def adjust_meter(self, action: Literal['increase', 'decrease', None], param: Literal['temperature', 'top_k', 'top_p'], state):
        """ Adjusts Agent's state. """

        if action is not None:
            if param == 'temperature':
                # self.temperature.update(action, state)

                if action == 'increase':
                    self.temperature.increase()
                else:
                    self.temperature.decrease()

            elif param == "top_k":
                # self.top_k.update(action, state)

                if action == 'increase':
                    self.top_k.increase()
                else:
                    self.top_k.decrease()
            else:
                # self.top_p.update(action, state)

                if action == 'increase':
                    self.top_p.increase()
                else:
                    self.top_p.decrease()

    def __post_init__(self):
        self.validate()

class Trial(ABC):
    def __init__(self):
        self.N = 3
        self.state = DecodeState()
        self.state_labels = ('temperature', 'top_k', 'top_p')
        self.action_labels = ('increase', 'decrease', 'NA')

    @abstractmethod
    def update(self, *args, **kwargs):
        """ Updates attributes with current observational space after an action is taken. """
        # for param, next_action in self.next_action(self.action_labels.index):
        # self.adjust_meter(action=next_action, param=param)
        # print('New state after trial run ', self.state)
        pass

    def next_action(self):
        """ Returns the decode param and next action based on the given state or current attributes e.g. Transitioning matrix. Next action to adjust the params - increase / decrease or no change. """
        pass

    @abstractmethod
    def run(self):
        """ Returns the decode parameters. """
        pass