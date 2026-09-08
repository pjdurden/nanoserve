"""Day 53 tests: the decode step's inputs, written into buffers that do not move.

Day 50 moved the addressing out of the forward, Day 51 made the read rectangle a
window on a persistent table, Day 52 closed the shape into a small bucket set. What
is left between here and `torch.cuda.graph` is on the input side. `positions`,
`write_slots` and `context_lens` are still built out of Python lists into fresh
tensors every step, and the engine's padded `input_ids` is a `torch.cat`. A replay
does not accept a fresh tensor: it re-runs kernels bound to the addresses they were
recorded against, so an input that is a new allocation every step is an input the
graph never sees.

This file allocates them once. `DecodeInputs` is four `[max_batch_size]` int64
buffers, a step writes cells into them and hands back windows, and the address the
forward is given on step 500 is the address it was given on step 1.

Four claims:

  1. **One allocation, then only writes.** The buffers are sized at construction
     and `check_addresses_stable` is the gate that a run of any length never
     replaced one. What a step costs is `rows` cells per buffer, not a tensor.
  2. **Every input the forward sees is a window.** `input_ids`, `positions`,
     `write_slots` and `context_lens` all come back as views on that storage, the
     way Day 51's rectangle already did.
  3. **Sharing the storage disarms two gates, and the snapshot is what re-arms
     them.** Day 51 kept `context_lens` a fresh copy on purpose: `check_plan_current`
     compares a plan's lengths against the cache's tables, so a length that lives in
     a buffer the next step overwrites always agrees with them. A shared plan
     therefore has to carry its own host-side copies, and `check_snapshots_present`
     is the gate that says it does.
  4. **One buffer set covers the whole capture list, and it was never the memory
     problem anyway.** A graph is per shape; a buffer is indexed by batch position,
     so every shape's window is a prefix of the same storage. The arithmetic that
     proves it also shows the input side of a decode step is kilobytes against the
     slot table's megabytes: these buffers exist for their address, not their size.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.buckets import check_pad_inert
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.compiled import DecodeShape
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine, Request
from nanoserve.inputs import (
    DECODE_INPUTS,
    INPUT_ITEMSIZE,
    DecodeInputs,
    InputBuffer,
    InputsUnsound,
    check_addresses_stable,
    check_inputs_fit,
    check_one_set_covers,
    check_plan_inputs_persistent,
    check_snapshots_present,
    check_step_inputs_persistent,
    fresh_allocations,
    fresh_cells,
    input_bytes,
    input_cells,
    per_shape_bytes,
    sharing_ratio,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import PlanUnsound, check_plan_addressing, check_plan_current, plan_decode
from nanoserve.slots import SlotsUnsound, check_window_intact, table_bytes


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model(seed: int = 0) -> tuple[LlamaModel, ModelConfig]:
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg)), cfg


def _kv(cfg: ModelConfig, batch: int, seq: int, seed: int = 1):
    torch.manual_seed(seed)
    shape = (batch, cfg.num_key_value_heads, seq, cfg.head_dim)
    return torch.randn(*shape), torch.randn(*shape)


def _prefilled(prompts, *, num_blocks=32, block_size=4, batch_size=None, seed=1, **kwargs):
    """A batched cache holding one prefill per row, ready to decode."""
    cfg = _tiny_config()
    batch_size = batch_size if batch_size is not None else len(prompts)
    cache = BatchedPagedKVCache(
        cfg,
        BlockAllocator(num_blocks=num_blocks, block_size=block_size),
        batch_size,
        **kwargs,
    )
    _prefill_rows(cache, cfg, prompts, tuple(range(len(prompts))), seed=seed)
    return cfg, cache


def _prefill_rows(cache, cfg, prompts, rows, seed=1):
    batch = pad_prompts(list(prompts), pad_id=0)
    for layer in range(cfg.num_hidden_layers):
        k, v = _kv(cfg, len(prompts), batch.max_length, seed=seed + layer)
        cache.write(layer, k, v, batch.attention_mask, rows=rows)


# --- what a buffer is ---------------------------------------------------------------


def test_a_buffer_allocates_its_whole_column_at_construction():
    """The point of the day: one allocation, and no per-step one after it."""
    buffer = InputBuffer("write_slots", 8)

    assert tuple(buffer.buffer.shape) == (8,)
    assert buffer.buffer.dtype == torch.long
    assert buffer.max_rows == 8


def test_a_buffer_with_a_tail_keeps_the_shape_the_forward_wants():
    """`input_ids` and `positions` are `[rows, 1]` and the buffer says so, so the
    window needs no reshape at the call site."""
    buffer = InputBuffer("positions", 4, tail=(1,))

    assert tuple(buffer.buffer.shape) == (4, 1)
    assert buffer.row_cells == 1


def test_a_buffer_refuses_a_batch_it_could_never_hold():
    with pytest.raises(ValueError, match="at least one row"):
        InputBuffer("write_slots", 0)


def test_a_buffer_refuses_a_tail_with_nothing_in_it():
    with pytest.raises(ValueError, match="at least one cell"):
        InputBuffer("write_slots", 4, tail=(0,))


def test_a_window_is_the_buffers_own_storage():
    buffer = InputBuffer("write_slots", 8)

    window = buffer.window(3)

    assert tuple(window.shape) == (3,)
    assert buffer.owns(window)


def test_a_window_past_the_end_of_the_buffer_is_refused():
    buffer = InputBuffer("write_slots", 4)

    with pytest.raises(ValueError, match="4"):
        buffer.window(5)


def test_a_window_of_no_rows_is_refused():
    buffer = InputBuffer("write_slots", 4)

    with pytest.raises(ValueError, match="at least one row"):
        buffer.window(0)


def test_a_write_lands_in_the_buffer_and_comes_back_as_a_window():
    buffer = InputBuffer("write_slots", 8)

    window = buffer.write([9, 7, 5])

    assert window.tolist() == [9, 7, 5]
    assert buffer.owns(window)
    assert buffer.buffer[:3].tolist() == [9, 7, 5]


def test_a_write_with_a_tail_fills_row_major():
    buffer = InputBuffer("positions", 4, tail=(1,))

    window = buffer.write([11, 12])

    assert window.tolist() == [[11], [12]]


def test_a_write_does_not_move_the_buffer():
    """The whole property a replay depends on, stated as an address."""
    buffer = InputBuffer("write_slots", 8)
    before = buffer.address

    for step in range(50):
        buffer.write([step, step + 1, step + 2])

    assert buffer.address == before


def test_a_write_leaves_the_rows_past_it_alone():
    buffer = InputBuffer("write_slots", 6)
    buffer.write([1, 2, 3, 4, 5, 6])

    buffer.write([9, 9])

    assert buffer.buffer.tolist() == [9, 9, 3, 4, 5, 6]


def test_a_write_wider_than_the_buffer_is_refused():
    buffer = InputBuffer("write_slots", 3)

    with pytest.raises(ValueError, match="3"):
        buffer.write([1, 2, 3, 4])


def test_an_empty_write_is_refused():
    buffer = InputBuffer("write_slots", 3)

    with pytest.raises(ValueError, match="at least one row"):
        buffer.write([])


def test_a_write_that_does_not_divide_into_rows_is_refused():
    buffer = InputBuffer("positions", 4, tail=(2,))

    with pytest.raises(ValueError, match="2 cells"):
        buffer.write([1, 2, 3])


def test_a_wider_window_than_the_write_keeps_what_the_buffer_held():
    """How a padded row gets its token. Nothing but a legal id is ever written
    here, so whatever is left over is legal, which is Day 51's argument for the
    slot table's padding arriving on a different buffer."""
    buffer = InputBuffer("input_ids", 4, tail=(1,))
    buffer.write([5, 6, 7, 8])

    window = buffer.write([1, 2], window=4)

    assert window.tolist() == [[1], [2], [7], [8]]


def test_a_window_narrower_than_the_write_is_refused():
    buffer = InputBuffer("write_slots", 8)

    with pytest.raises(ValueError, match="window"):
        buffer.write([1, 2, 3], window=2)


def test_a_tensor_write_copies_into_the_buffer():
    """The device-side path: last step's sampled tokens are already on the device,
    so they are copied straight in and never go home."""
    buffer = InputBuffer("input_ids", 4, tail=(1,))
    tokens = torch.tensor([3, 4], dtype=torch.long)

    window = buffer.write(tokens)

    assert window.tolist() == [[3], [4]]
    assert buffer.owns(window)


def test_a_tensor_write_is_a_copy_and_not_an_alias():
    """Day 47 handed the forward a view on the sampler's output. A replay cannot
    use that: the address is the sampler's and it is new every step. The buffer
    turns a zero-copy view into a one-copy write, which is the price of an address."""
    buffer = InputBuffer("input_ids", 4, tail=(1,))
    tokens = torch.tensor([3, 4], dtype=torch.long)
    window = buffer.write(tokens)

    tokens[0] = 99

    assert window.tolist() == [[3], [4]]
    assert not buffer.owns(tokens)


def test_a_tensor_of_the_wrong_size_is_refused():
    buffer = InputBuffer("input_ids", 4, tail=(1,))

    with pytest.raises(ValueError, match="5"):
        buffer.write(torch.arange(5))


def test_writes_are_counted_in_calls_and_in_cells():
    buffer = InputBuffer("write_slots", 8)

    buffer.write([1, 2, 3])
    buffer.write([4, 5])

    assert buffer.writes == 2
    assert buffer.written_cells == 5


def test_a_host_write_and_a_device_write_are_counted_apart():
    buffer = InputBuffer("input_ids", 4, tail=(1,))

    buffer.write([1, 2])
    buffer.write(torch.tensor([3, 4], dtype=torch.long))

    assert (buffer.staged_writes, buffer.device_writes) == (1, 1)


def test_a_cpu_buffer_stages_in_its_own_storage():
    """There is nothing to transfer, so the staging tensor is the buffer itself and
    a write is one numpy assignment. On CUDA they are two tensors and the write ends
    in a pinned host-to-device copy."""
    buffer = InputBuffer("write_slots", 4)

    assert buffer.staging is buffer.buffer


def test_moving_to_the_same_device_changes_nothing():
    buffer = InputBuffer("write_slots", 4)
    before = buffer.address

    buffer.to(None)
    buffer.to("cpu")

    assert buffer.address == before
    assert buffer.moves == 0


def test_a_buffer_does_not_own_a_stranger():
    buffer = InputBuffer("write_slots", 4)

    assert not buffer.owns(torch.zeros(4, dtype=torch.long))


def test_a_buffer_renders_its_shape_and_its_price():
    line = InputBuffer("positions", 8, tail=(1,)).render()

    assert "positions" in line
    assert "8" in line


# --- the set of them ----------------------------------------------------------------


def test_decode_inputs_holds_one_buffer_per_input_the_forward_takes():
    inputs = DecodeInputs(8)

    assert tuple(b.name for b in inputs.buffers) == tuple(n for n, _ in DECODE_INPUTS)


def test_the_four_buffers_have_the_shapes_the_forward_wants():
    inputs = DecodeInputs(8)

    assert tuple(inputs.input_ids.buffer.shape) == (8, 1)
    assert tuple(inputs.positions.buffer.shape) == (8, 1)
    assert tuple(inputs.write_slots.buffer.shape) == (8,)
    assert tuple(inputs.context_lens.buffer.shape) == (8,)


def test_setting_each_input_hands_back_a_window_on_its_own_buffer():
    inputs = DecodeInputs(8)

    assert inputs.owns(inputs.set_input_ids([1, 2]))
    assert inputs.owns(inputs.set_positions([3, 4]))
    assert inputs.owns(inputs.set_write_slots([5, 6]))
    assert inputs.owns(inputs.set_context_lens([7, 8]))


def test_the_four_buffers_are_four_allocations():
    """Not one tensor sliced four ways: each input is written and read on its own,
    and a fused buffer would make one write's stride the next one's problem."""
    inputs = DecodeInputs(8)

    assert len(set(inputs.addresses)) == 4


def test_the_addresses_are_the_same_after_a_hundred_steps():
    inputs = DecodeInputs(4)
    before = inputs.addresses

    for step in range(100):
        inputs.set_input_ids([step % 7, 1])
        inputs.set_positions([step, step])
        inputs.set_write_slots([0, 1])
        inputs.set_context_lens([step + 1, step + 1])

    assert inputs.addresses == before
    check_addresses_stable(inputs, before)


def test_a_step_costs_cells_and_not_allocations():
    inputs = DecodeInputs(4)

    for _ in range(10):
        inputs.set_input_ids([1, 2])
        inputs.set_positions([3, 4])
        inputs.set_write_slots([5, 6])
        inputs.set_context_lens([7, 8])

    assert inputs.writes == 40
    assert inputs.written_cells == 80


def test_inputs_refuse_a_batch_bigger_than_they_were_sized_for():
    inputs = DecodeInputs(2)

    with pytest.raises(ValueError, match="2"):
        inputs.set_write_slots([1, 2, 3])


def test_inputs_report_what_they_weigh():
    inputs = DecodeInputs(8)

    assert inputs.cells == input_cells(8)
    assert inputs.bytes == input_bytes(8)


def test_inputs_render_one_line_for_a_log():
    line = DecodeInputs(8).render()

    assert "8" in line
    assert "bytes" in line


def test_moving_the_set_moves_every_buffer_in_it():
    inputs = DecodeInputs(4)
    before = inputs.addresses

    inputs.to("cpu")

    assert inputs.addresses == before


# --- the arithmetic -----------------------------------------------------------------


def test_input_cells_is_one_column_per_buffer():
    assert input_cells(8) == 8 * len(DECODE_INPUTS)


def test_input_bytes_is_int64_because_these_are_index_tensors():
    assert input_bytes(8) == input_cells(8) * INPUT_ITEMSIZE


def test_fresh_cells_is_what_the_per_step_build_used_to_write():
    assert fresh_cells(steps=100, rows=4) == 100 * 4 * len(DECODE_INPUTS)


def test_fresh_allocations_is_one_tensor_per_input_per_step():
    assert fresh_allocations(steps=100) == 100 * len(DECODE_INPUTS)


def test_the_cost_functions_refuse_nonsense():
    with pytest.raises(ValueError, match="at least one row"):
        input_cells(0)
    with pytest.raises(ValueError, match="non-negative"):
        fresh_cells(steps=-1, rows=4)
    with pytest.raises(ValueError, match="at least one row"):
        fresh_cells(steps=1, rows=0)
    with pytest.raises(ValueError, match="non-negative"):
        fresh_allocations(steps=-1)
    with pytest.raises(ValueError, match="at least one byte"):
        input_bytes(8, itemsize=0)


def test_one_buffer_set_covers_every_shape_in_a_capture_list():
    """The day's second question, answered. A graph is recorded per shape; a buffer
    is indexed by batch position, so every shape's window is a prefix of the same
    storage and the capture list needs one set, not `count` of them."""
    shapes = (DecodeShape(rows=1, context_width=128), DecodeShape(rows=4, context_width=256))

    check_one_set_covers(shapes, DecodeInputs(4))


def test_a_shape_wider_than_the_buffers_is_a_set_that_does_not_cover():
    shapes = (DecodeShape(rows=8, context_width=128),)

    with pytest.raises(InputsUnsound, match="8 rows"):
        check_one_set_covers(shapes, DecodeInputs(4))


def test_per_shape_bytes_is_what_a_buffer_set_per_graph_would_cost():
    shapes = (DecodeShape(rows=1, context_width=128), DecodeShape(rows=4, context_width=256))

    assert per_shape_bytes(shapes) == (1 + 4) * len(DECODE_INPUTS) * INPUT_ITEMSIZE


def test_sharing_one_set_is_cheaper_than_a_set_per_shape():
    shapes = tuple(
        DecodeShape(rows=r, context_width=w) for r in (1, 2, 4) for w in (128, 256, 512)
    )

    assert sharing_ratio(shapes, 4) > 1


def test_the_whole_input_side_of_a_decode_step_is_kilobytes():
    """The finding the arithmetic was supposed to produce and did not. The inputs
    are four `[max_batch]` vectors, so at serving size they are 8 KB against the
    slot table's 16 MB: these buffers are worth having for their address and not
    for their size, and a capture list that needed one set per shape would still
    not be where the memory went."""
    assert input_bytes(256) < 16_384
    assert input_bytes(256) * 1000 < table_bytes(256, 8192)


# --- the plan writes into them ------------------------------------------------------


def test_a_cache_without_persistent_inputs_is_untouched():
    _, cache = _prefilled([[1, 2, 3], [4]])

    assert cache.decode_inputs is None
    assert not plan_decode(cache).context_snapshot


def test_a_planned_step_writes_its_addressing_into_the_buffers():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)

    plan = plan_decode(cache)

    inputs = cache.decode_inputs
    assert inputs.owns(plan.positions)
    assert inputs.owns(plan.write_slots)
    assert inputs.owns(plan.context_lens)
    check_plan_inputs_persistent(plan, inputs)


def test_a_planned_step_over_buffers_says_the_same_thing_it_always_did():
    """The claim that matters: this is a change of where the numbers live."""
    _, plain = _prefilled([[1, 2, 3], [4]])
    _, buffered = _prefilled([[1, 2, 3], [4]], persist_inputs=True)

    want, got = plan_decode(plain), plan_decode(buffered)

    assert got.positions.tolist() == want.positions.tolist()
    assert got.write_slots.tolist() == want.write_slots.tolist()
    assert got.context_lens.tolist() == want.context_lens.tolist()
    check_plan_addressing(got)


def test_two_steps_reuse_the_same_storage():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    before = cache.decode_inputs.addresses

    plan_decode(cache)
    plan_decode(cache)

    check_addresses_stable(cache.decode_inputs, before)


def test_the_second_step_overwrites_the_first_steps_lengths():
    """The hazard the snapshot exists for, stated as an observation rather than as
    a gate: a plan's `context_lens` is storage the next step writes through."""
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    first = plan_decode(cache)

    plan_decode(cache)

    assert first.context_lens.tolist() == [5, 3]
    assert first.context_snapshot == (4, 2)


def test_the_rectangle_is_still_a_window_on_the_slot_table():
    """Day 51's buffer and Day 53's are different buffers and both are windows."""
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)

    plan = plan_decode(cache)

    assert cache.slot_table.is_window(plan.slot_mapping)
    assert not cache.decode_inputs.owns(plan.slot_mapping)


def test_the_lengths_no_longer_cost_a_tensor_a_step():
    """What Day 51 left on the table. `SlotTable.read` built a fresh `[rows]` int64
    every step; with a writer it puts them where the caller already has room."""
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)

    plan = plan_decode(cache)

    assert cache.decode_inputs.context_lens.owns(plan.context_lens)


def test_buckets_and_buffers_compose():
    _, cache = _prefilled(
        [[1, 2, 3], [4]], batch_size=4, persist_inputs=True, bucket_decode=True
    )

    plan = plan_decode(cache, rows=(0, 1))

    assert plan.graph_rows == 2
    check_pad_inert(plan)
    check_plan_inputs_persistent(plan, cache.decode_inputs)


def test_a_padded_rows_addressing_is_written_and_not_inherited():
    """`input_ids` may keep whatever the buffer held, because any token id is legal
    for a row nobody reads. `write_slots` and `context_lens` may not: the write is
    an `index_put` over the padded batch, so a stale slot there is a made-up token
    landing in a real sequence's history."""
    _, cache = _prefilled(
        [[1, 2, 3], [4], [5, 6]], batch_size=4, persist_inputs=True, bucket_decode=True
    )
    cache.decode_inputs.set_write_slots([31, 31, 31, 31])
    cache.decode_inputs.set_context_lens([9, 9, 9, 9])

    plan = plan_decode(cache, rows=(0, 1, 2))

    assert plan.pad_rows == 1
    assert plan.write_slots.tolist()[3:] == [cache.sink_slot]
    assert plan.context_lens.tolist()[3:] == [0]
    check_pad_inert(plan)


# --- the gates the sharing disarmed -------------------------------------------------


def test_the_shared_lengths_would_pass_a_stale_plan_on_their_own():
    """Day 51 kept `context_lens` a fresh copy for exactly this reason and this is
    the same argument arriving from the other side: a buffer the next step writes
    through always agrees with the tables, so the tensor stops being able to fail."""
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    plan = plan_decode(cache)
    plan_decode(cache)

    assert plan.context_lens.tolist() == [n.num_tokens for n in cache.tables]

    with pytest.raises(PlanUnsound, match="stale"):
        check_plan_current(plan, cache)


def test_the_staleness_gate_reads_the_snapshot_when_there_is_one():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    plan = plan_decode(cache)

    check_plan_current(plan, cache)


def test_a_plan_with_no_snapshot_falls_back_to_its_own_tensor():
    """Day 50's and Day 51's plans are unchanged: nothing shares their storage, so
    the tensor is still the witness and `check_plan_current` still reads it."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    plan_decode(cache)

    with pytest.raises(PlanUnsound, match="stale"):
        check_plan_current(plan, cache)


def test_check_window_intact_uses_the_snapshot_when_the_slots_are_shared():
    """The second disarmed gate. Its witnesses were `write_slots` and
    `context_lens`, which used to be copies taken at build time and are now storage
    the next step writes through."""
    cfg, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    plan = plan_decode(cache)
    check_window_intact(plan)

    cache.reset_row(1)
    _prefill_rows(cache, cfg, [[7, 8]], (1,), seed=9)
    plan_decode(cache)

    with pytest.raises(SlotsUnsound, match="row 1"):
        check_window_intact(plan)


def test_check_snapshots_present_refuses_a_shared_plan_that_carries_none():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    plan = plan_decode(cache)

    stripped = replace(plan, context_snapshot=(), write_snapshot=())

    with pytest.raises(InputsUnsound, match="snapshot"):
        check_snapshots_present(stripped, cache.decode_inputs)


def test_check_snapshots_present_accepts_a_plan_that_carries_both():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)

    check_snapshots_present(plan_decode(cache), cache.decode_inputs)


def test_check_snapshots_present_ignores_a_plan_that_shares_nothing():
    _, cache = _prefilled([[1, 2, 3], [4]])

    check_snapshots_present(plan_decode(cache), DecodeInputs(4))


def test_a_snapshot_that_disagrees_with_its_own_tensor_is_refused():
    """A snapshot is a copy and a copy can be wrong. Taken at build time off the
    same list the tensor was written from, it cannot be, so a disagreement means
    somebody built one of the two by hand."""
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    plan = plan_decode(cache)

    lying = replace(plan, context_snapshot=(1, 1))

    with pytest.raises(InputsUnsound, match="snapshot"):
        check_snapshots_present(lying, cache.decode_inputs)


def test_check_plan_inputs_persistent_catches_a_freshly_built_tensor():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    plan = plan_decode(cache)

    fresh = replace(plan, positions=plan.positions.clone())

    with pytest.raises(InputsUnsound, match="positions"):
        check_plan_inputs_persistent(fresh, cache.decode_inputs)


def test_check_step_inputs_persistent_covers_the_tokens_too():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    plan = plan_decode(cache)
    ids = cache.decode_inputs.set_input_ids([1, 2])

    check_step_inputs_persistent(ids, plan, cache.decode_inputs)

    with pytest.raises(InputsUnsound, match="input_ids"):
        check_step_inputs_persistent(ids.clone(), plan, cache.decode_inputs)


def test_check_addresses_stable_catches_a_reallocation():
    inputs = DecodeInputs(4)
    before = inputs.addresses
    inputs.write_slots.buffer = torch.zeros(4, dtype=torch.long)

    with pytest.raises(InputsUnsound, match="write_slots"):
        check_addresses_stable(inputs, before)


def test_check_addresses_stable_refuses_a_snapshot_of_the_wrong_size():
    with pytest.raises(ValueError, match="4"):
        check_addresses_stable(DecodeInputs(4), (0, 0))


def test_check_inputs_fit_accepts_a_batch_the_buffers_hold():
    check_inputs_fit(4, DecodeInputs(4))


def test_check_inputs_fit_refuses_one_they_do_not():
    with pytest.raises(InputsUnsound, match="5 rows"):
        check_inputs_fit(5, DecodeInputs(4))


# --- the engine ---------------------------------------------------------------------


def _run(engine, requests):
    for request in requests:
        engine.add_request(request)
    done = {}
    while engine.has_unfinished():
        out = engine.step()
        for request in out.finished:
            done[request.request_id] = list(request.output_token_ids)
    return done


def test_an_engine_without_persistent_inputs_is_untouched():
    model, _ = _model()
    engine = Engine.build(model, num_blocks=32, block_size=4, max_batch_size=4)

    assert engine.cache.decode_inputs is None


def test_an_engine_over_persistent_inputs_generates_the_same_tokens():
    model, _ = _model()
    out = []
    for persist in (False, True):
        engine = Engine.build(
            model,
            num_blocks=64,
            block_size=4,
            max_batch_size=4,
            max_model_len=64,
            persist_inputs=persist,
        )
        out.append(
            _run(
                engine,
                [
                    Request("a", [1, 2, 3, 4], max_new_tokens=10),
                    Request("b", [5, 6], max_new_tokens=10),
                ],
            )
        )
    assert out[0] == out[1]


def test_an_engine_over_buffers_and_buckets_generates_the_same_tokens():
    """The three days compose or none of them ship: a bucketed step pads the batch,
    a persistent step writes the padding into a fixed address, and the text does not
    move."""
    model, _ = _model()
    out = []
    for extra in ({}, {"persist_inputs": True, "bucket_decode": True}):
        engine = Engine.build(
            model,
            num_blocks=64,
            block_size=4,
            max_batch_size=4,
            max_model_len=64,
            **extra,
        )
        out.append(
            _run(
                engine,
                [
                    Request("a", [1, 2, 3, 4], max_new_tokens=10),
                    Request("b", [5, 6], max_new_tokens=10),
                ],
            )
        )
    assert out[0] == out[1]


def test_a_whole_run_never_reallocates_an_input():
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=64,
        block_size=4,
        max_batch_size=4,
        max_model_len=64,
        persist_inputs=True,
    )
    before = engine.cache.decode_inputs.addresses

    _run(engine, [Request("a", [1, 2, 3, 4], max_new_tokens=16)])

    check_addresses_stable(engine.cache.decode_inputs, before)
    assert engine.cache.decode_inputs.writes > 16
