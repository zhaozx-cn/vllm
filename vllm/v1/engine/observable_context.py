from dataclasses import dataclass
from typing import Optional

import itertools

from vllm.transformers_utils.detokenizer_utils import convert_ids_list_to_tokens
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.v1.engine import EngineCoreEvent, EngineCoreRequest, EngineCoreOutput
from vllm.v1.outputs import IterStats

NONES = itertools.repeat(None)

@dataclass
class ObservableContext:

    iter_batch_size: list[int]
    iter_waiting_size: list[int]
    iter_total_tokens_count: list[int]
    scheduled_time: list[float]
    token_time: list[float]
    candidate_token_ids: Optional[list[list[int]]]
    candidate_decoded_tokens: Optional[list[list[str]]]
    candidate_token_probs: Optional[list[list[float]]]
    tokenizer: Optional[AnyTokenizer]
    events: Optional[list[EngineCoreEvent]]
    num_cached_tokens: int = None
    not_empty: bool = False

    @classmethod
    def from_new_request(cls, tokenizer: Optional[AnyTokenizer]):
            return cls(
                iter_batch_size=[],
                iter_waiting_size=[],
                iter_total_tokens_count=[],
                scheduled_time=[],
                token_time=[],
                candidate_token_ids=[],
                candidate_decoded_tokens=[],
                candidate_token_probs=[],
                tokenizer=tokenizer,
                events=[],
            )

    def _update_iter_stats(self, iter_stats: IterStats, new_token_ids: list[int]) -> None:
        if not new_token_ids:
            return

        self.not_empty = True
        new_tokens_num = len(new_token_ids)
        self.iter_total_tokens_count.extend([iter_stats.iter_total_tokens_count] * new_tokens_num)
        self.scheduled_time.extend([iter_stats.token_scheduled_time] * new_tokens_num)
        self.token_time.extend([iter_stats.token_output_time] * new_tokens_num)
        self.iter_batch_size.extend([iter_stats.iter_batch_size] * new_tokens_num)
        self.iter_waiting_size.extend([iter_stats.iter_waiting_size] * new_tokens_num)
        if self.num_cached_tokens is None:
            self.num_cached_tokens = iter_stats.num_cached_tokens

        
        if iter_stats.logprobs_tensors_for_trace:
            token_ids_lst, logprobs_lst, ranks_lst = iter_stats.logprobs_tensors_for_trace
            for _, logprobs, token_ids in zip(ranks_lst, logprobs_lst, token_ids_lst):
                # Detokenize (non-incrementally).
                decoded_tokens = NONES if self.tokenizer is None else (
                    convert_ids_list_to_tokens(self.tokenizer, token_ids, True))
                # Update with the Logprob dictionary for this pos.
                self.candidate_token_ids.append(token_ids)
                self.candidate_decoded_tokens.append(decoded_tokens)
                self.candidate_token_probs.append(logprobs)
        else:
            self.candidate_token_ids.extend([[id,id] for id in new_token_ids])
            decoded_tokens = NONES if self.tokenizer is None else (
                    convert_ids_list_to_tokens(self.tokenizer, new_token_ids, True))
            self.candidate_decoded_tokens.extend([[token,token] for token in decoded_tokens])

    def _update_events(self, events: list[EngineCoreEvent]) -> None:
        self.events.extend(events)

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.iter_stats is not None:
            self._update_iter_stats(output.iter_stats, output.new_token_ids)
        if output.events is not None:
            self._update_events(output.events)
