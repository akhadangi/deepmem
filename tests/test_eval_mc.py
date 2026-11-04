import torch
from deepmem.eval_mc import _encode_candidate, candidates_and_answer, score_candidates, perturb_text
from deepmem.model import LosslessSubbandByteLadderConfig, LosslessSubbandByteLadderLM


def test_score_length():
    m = LosslessSubbandByteLadderLM(LosslessSubbandByteLadderConfig(
        block_size=32, min_length=8, d_model=32, max_width=32, max_heads=4, n_heads=4,
        n_initial_layers=0, n_final_layers=0, custom_sem_widths=[32, 32, 32],
        down_depths=[0, 0], up_refine_depths=[0, 0], disable_multiscale=True,
        bottleneck_shared_state=False, lift_rank=8, carrier_proj_rank=8)).eval()
    prompt = 'Hello:'
    options = [' 2', ' 3']
    values = score_candidates(m, prompt, options, precision='none', length_norm=False)
    assert len(values) == 2
    assert all(torch.isfinite(torch.tensor(values)))
    ids, start = _encode_candidate(prompt, options[0], 32)
    assert ids.size(0) == 32 and start == 30


def test_dataset_fields():
    prompt, options, answer = candidates_and_answer({'goal':'g', 'sol1':'x', 'sol2':'y', 'label':1}, 'piqa')
    assert answer == 1 and options == ['x', 'y']
    prompt, options, answer = candidates_and_answer({'question':'q', 'choices':{'text':['a','b'], 'label':['A','B']}, 'answerKey':'B'}, 'arc_easy')
    assert answer == 1
