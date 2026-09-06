"""Day 51 tests: the read rectangle stops being rebuilt and starts being a window.

Day 50 moved a decode step's addressing out of the forward and priced what that
cost. The `[rows, max_ctx]` read rectangle is int64 built from scratch on the host
every step, and `max_ctx` grows by one per step, so 512 steps over 4 rows write
558,080 cells to append 2,048. `check_rebuild_bounded` is the gate that says when
that stops being affordable, and it fails a 512-step run by a factor of eight.

This file is the fix. A `SlotTable` is one `[max_batch_size, max_model_len]` int64
buffer allocated once. A decode step writes exactly one cell per row into it, and
the rectangle the forward reads is `slots[:rows, :width]`: a *window* on that
buffer, the same storage at the same address, not a fresh tensor. Host work per
step goes from `rows * max_ctx` to `rows`, and the rectangle stops being an
allocation at all.

Four claims, and the third is the one that cost the day:

  1. **The window says exactly what the rebuild said.** `rebuild_mapping` is the
     Day-50 construction, extracted, and it is both the resync path and the oracle:
     the table and a fresh rebuild agree after every step of a real run.
  2. **The table is a mirror, not a second source of truth.** `BlockTable` still
     owns where a token lives. The table copies it, catches up on any row a prefill
     or a reset moved (`sync`), and `check_slots_agree` is the gate that the copy
     has not drifted.
  3. **A window is live storage, and one of the two tensors could not be one.**
     Appends write at column `length`, which is past a held window's real region,
     so the rectangle a plan holds keeps saying what it said. The *lengths* are the
     opposite: an append writes the very cell a previous plan is holding, so a
     `context_lens` that tracked the table would silently disarm Day 50's staleness
     gate, which compares a plan's lengths against the cache's. They are `[rows]`
     and linear, so copying them costs nothing and keeps the gate armed. One buffer
     is safe to share and the other is not, and the difference is whether a later
     step writes inside the window or past its edge.
  4. **The fixed allocation is the new price.** `[max_batch_size, max_model_len]`
     int64 is a real number of bytes reserved forever: 256 rows at 131,072 tokens
     is 268 MB of addressing. vLLM keeps block ids rather than slots, which is the
     same table `block_size` times smaller, and `block_table_bytes` is that
     arithmetic. `check_table_fits` is the gate.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator, BlockTable
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import PlanUnsound, check_plan_current, plan_decode, rebuild_cells
from nanoserve.slots import (
    DEFAULT_RESYNC_LIMIT,
    SlotsUnsound,
    SlotTable,
    SlotTableFull,
    block_table_bytes,
    block_table_cells,
    check_appends_incremental,
    check_mapping_is_window,
    check_resyncs_bounded,
    check_slots_agree,
    check_table_fits,
    check_window_intact,
    incremental_cells,
    rebuild_mapping,
    resident_share,
    table_bytes,
    table_cells,
)


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


def _prefilled(prompts, *, num_blocks=32, block_size=4, batch_size=None, seed=1):
    """A batched cache holding one prefill per row, ready to decode."""
    cfg = _tiny_config()
    batch_size = batch_size if batch_size is not None else len(prompts)
    cache = BatchedPagedKVCache(
        cfg, BlockAllocator(num_blocks=num_blocks, block_size=block_size), batch_size
    )
    _prefill_rows(cache, cfg, prompts, tuple(range(len(prompts))), seed=seed)
    return cfg, cache


def _prefill_rows(cache, cfg, prompts, rows, seed=1):
    """Write one prefill per named row, the ragged key-mask path."""
    batch = pad_prompts(list(prompts), pad_id=0)
    for layer in range(cfg.num_hidden_layers):
        k, v = _kv(cfg, len(prompts), batch.max_length, seed=seed + layer)
        cache.write(layer, k, v, batch.attention_mask, rows=rows)


def _block_table(num_tokens: int, *, num_blocks=8, block_size=2) -> BlockTable:
    table = BlockTable(BlockAllocator(num_blocks=num_blocks, block_size=block_size))
    table.append(num_tokens)
    return table


# --- what a slot table is ---------------------------------------------------------


def test_a_slot_table_allocates_its_whole_rectangle_at_construction():
    """The point of the day: one allocation, and no per-step one after it."""
    table = SlotTable(4, 16)

    assert tuple(table.slots.shape) == (4, 16)
    assert table.slots.dtype == torch.long
    assert (table.max_batch_size, table.max_model_len) == (4, 16)


def test_a_fresh_table_holds_no_tokens_and_reads_as_zeros():
    table = SlotTable(3, 8)

    assert [table.length(r) for r in range(3)] == [0, 0, 0]
    assert table.slots.sum().item() == 0


def test_a_table_prices_itself_in_bytes():
    table = SlotTable(4, 16)

    assert table.cells == 64
    assert table.bytes == 64 * 8


def test_a_table_needs_a_positive_shape():
    with pytest.raises(ValueError, match="at least one row"):
        SlotTable(0, 16)
    with pytest.raises(ValueError, match="at least one token"):
        SlotTable(4, 0)


def test_moving_a_table_to_the_device_it_is_already_on_is_the_same_buffer():
    """`to` is a reallocation, so the no-op case has to actually be one."""
    table = SlotTable(2, 8)
    before = table.slots.data_ptr()

    assert table.to(None) is table
    assert table.to("cpu") is table
    assert table.slots.data_ptr() == before


# --- appending --------------------------------------------------------------------


def test_appending_writes_one_entry_per_row_at_that_rows_length():
    table = SlotTable(3, 8)
    table.append((0, 2), [5, 9])
    table.append((0, 2), [6, 10])

    assert table.slots[0, :2].tolist() == [5, 6]
    assert table.slots[2, :2].tolist() == [9, 10]
    assert [table.length(r) for r in range(3)] == [2, 0, 2]


def test_appending_leaves_the_rows_it_did_not_name_alone():
    table = SlotTable(3, 8)
    table.append((1,), [7])

    assert table.length(0) == 0 and table.length(2) == 0
    assert table.slots[0].sum().item() == 0


def test_appending_counts_one_cell_per_row_per_step():
    """The whole claim of the day, as a counter: host work is O(rows), not O(cells)."""
    table = SlotTable(4, 32)
    for step in range(6):
        table.append((0, 1, 2, 3), [step * 4 + r for r in range(4)])

    assert table.appended_cells == 24
    assert table.resynced_cells == 0


def test_appending_past_the_capacity_is_refused():
    table = SlotTable(2, 2)
    table.append((0,), [1])
    table.append((0,), [2])

    with pytest.raises(SlotTableFull, match="holds 2 tokens"):
        table.append((0,), [3])


def test_appending_needs_exactly_one_slot_per_row():
    table = SlotTable(3, 8)

    with pytest.raises(ValueError, match="one slot per row"):
        table.append((0, 1), [5])


def test_appending_to_a_row_the_table_does_not_have_is_refused():
    table = SlotTable(2, 8)

    with pytest.raises(ValueError, match="out of range"):
        table.append((2,), [5])


def test_a_row_cannot_be_named_twice_in_one_step():
    table = SlotTable(3, 8)

    with pytest.raises(ValueError, match="distinct"):
        table.append((1, 1), [5, 6])


# --- the mirror catching up --------------------------------------------------------


def test_a_resync_copies_a_block_tables_slots_into_the_row():
    block_table = _block_table(3)
    table = SlotTable(2, 8)
    table.resync(block_table, 1)

    assert table.length(1) == 3
    assert table.slots[1, :3].tolist() == [block_table.slot(p) for p in range(3)]
    assert table.resyncs == 1
    assert table.resynced_cells == 3


def test_a_sync_only_touches_the_rows_whose_length_disagrees():
    tables = [_block_table(3), _block_table(2)]
    table = SlotTable(2, 8)
    table.sync(tables, (0, 1))
    table.sync(tables, (0, 1))  # nothing moved, so nothing is rewritten

    assert table.resyncs == 2
    assert table.resynced_cells == 5


def test_a_sync_of_a_row_that_grew_rewrites_the_whole_row():
    """A prefill grows a block table by many tokens; the mirror takes all of them."""
    block_table = _block_table(2)
    table = SlotTable(2, 8)
    table.sync([block_table], (0,))
    block_table.append(3)
    table.sync([block_table], (0,))

    assert table.length(0) == 5
    assert table.slots[0, :5].tolist() == [block_table.slot(p) for p in range(5)]
    assert table.resyncs == 2


def test_resetting_a_row_empties_it_without_clearing_the_buffer():
    """Nothing is zeroed: what is past a row's length is padding, and padding is
    only ever required to be a legal index."""
    table = SlotTable(2, 8)
    table.append((0,), [7])
    table.reset(0)

    assert table.length(0) == 0
    assert table.slots[0, 0].item() == 7


def test_a_resync_past_the_capacity_is_refused():
    table = SlotTable(2, 4)
    block_table = _block_table(6, num_blocks=8, block_size=2)

    with pytest.raises(SlotTableFull, match="6 tokens"):
        table.resync(block_table, 0)


# --- reading it back ---------------------------------------------------------------


def test_a_contiguous_prefix_reads_back_as_a_window_on_the_buffer():
    """Same storage, same address: the rectangle is not built, it is pointed at."""
    table = SlotTable(4, 8)
    table.append((0, 1), [5, 9])
    mapping, lens = table.read((0, 1), 1)

    assert table.is_window(mapping)
    assert mapping.data_ptr() == table.slots.data_ptr()
    assert mapping.tolist() == [[5], [9]]
    assert lens.tolist() == [1, 1]


def test_a_window_is_a_stride_and_not_a_copy():
    """`[B, L]` narrowed to `[rows, width]` keeps the buffer's row stride."""
    table = SlotTable(4, 8)
    table.append((0, 1, 2), [1, 2, 3])
    mapping, _ = table.read((0, 1, 2), 1)

    assert mapping.stride() == (8, 1)
    assert not mapping.is_contiguous()


def test_a_scattered_row_selection_is_a_copy_and_says_so():
    """The gotcha. A view needs the rows the forward covers to be a prefix of the
    table's, and a scheduler hands out whichever slots are free."""
    table = SlotTable(4, 8)
    table.append((0, 1, 2), [5, 9, 3])
    mapping, lens = table.read((0, 2), 1)

    assert not table.is_window(mapping)
    assert mapping.tolist() == [[5], [3]]
    assert lens.tolist() == [1, 1]


def test_the_table_counts_windows_against_gathers():
    table = SlotTable(4, 8)
    table.append((0, 1, 2), [5, 9, 3])
    table.read((0, 1), 1)
    table.read((0, 1, 2), 1)
    table.read((0, 2), 1)

    assert (table.windows, table.gathers) == (2, 1)
    assert table.window_share == pytest.approx(2 / 3)


def test_a_read_wider_than_a_row_is_padded_with_whatever_the_buffer_holds():
    table = SlotTable(2, 8)
    table.append((0, 1), [5, 9])
    table.append((0,), [6])
    mapping, lens = table.read((0, 1), 2)

    assert mapping.tolist() == [[5, 6], [9, 0]]
    assert lens.tolist() == [2, 1]


def test_the_padding_of_a_reused_row_is_the_last_tenants_slot():
    """Not zero any more, and it never had to be: a padded cell is dereferenced and
    then masked away by `context_lens`, so all it owes anyone is being in range."""
    table = SlotTable(2, 8)
    table.append((0,), [5])
    table.append((0,), [6])
    table.reset(0)
    table.append((0,), [7])
    mapping, lens = table.read((0,), 2)

    assert mapping.tolist() == [[7, 6]]
    assert lens.tolist() == [1]


def test_a_read_narrower_than_a_rows_history_is_refused():
    table = SlotTable(2, 8)
    table.append((0,), [5])
    table.append((0,), [6])

    with pytest.raises(ValueError, match="holds 2 tokens"):
        table.read((0,), 1)


def test_a_read_wider_than_the_table_is_refused():
    table = SlotTable(2, 4)
    table.append((0,), [5])

    with pytest.raises(ValueError, match="at most 4"):
        table.read((0,), 5)


def test_a_read_of_no_rows_is_refused():
    table = SlotTable(2, 4)

    with pytest.raises(ValueError, match="at least one row"):
        table.read((), 1)


def test_a_table_renders_one_line_for_a_log():
    table = SlotTable(4, 16)
    table.append((0, 1), [1, 2])
    line = table.render()

    assert "4 rows x 16" in line
    assert "512 bytes" in line


# --- the extracted Day-50 construction ----------------------------------------------


def test_rebuild_mapping_is_the_day_50_rectangle():
    tables = [_block_table(3), _block_table(1)]
    mapping, lens = rebuild_mapping(tables, (0, 1))

    assert mapping.tolist() == [
        [tables[0].slot(p) for p in range(3)],
        [tables[1].slot(0), 0, 0],
    ]
    assert lens.tolist() == [3, 1]


def test_rebuilding_over_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing is cached"):
        rebuild_mapping([_block_table(0)], (0,))


def test_the_table_and_a_fresh_rebuild_agree_after_every_step():
    """The oracle. One is O(rows) a step and the other is O(cells), and they say
    the same thing."""
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    for _ in range(4):
        plan = plan_decode(cache)
        mapping, lens = rebuild_mapping(cache.tables, plan.rows)
        assert plan.slot_mapping.tolist() == mapping.tolist()
        assert plan.context_lens.tolist() == lens.tolist()


# --- wired into the cache -------------------------------------------------------------


def test_a_cache_sizes_its_table_from_the_pool_when_nothing_says_otherwise():
    """A row can never hold more tokens than the whole pool has slots, so the pool
    is the bound that always holds. It is a bound and not a plan: a real server
    passes `max_model_len` and gets a table three orders of magnitude smaller."""
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), 3)

    assert cache.max_model_len == 16
    assert tuple(cache.slot_table.slots.shape) == (3, 16)


def test_a_cache_takes_an_explicit_max_model_len():
    cfg = _tiny_config()
    allocator = BlockAllocator(num_blocks=64, block_size=16)
    cache = BatchedPagedKVCache(cfg, allocator, 3, max_model_len=128)

    assert cache.max_model_len == 128
    assert tuple(cache.slot_table.slots.shape) == (3, 128)


def test_a_planned_decode_hands_the_forward_a_window():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    assert cache.slot_table.is_window(plan.slot_mapping)
    check_mapping_is_window(cache.slot_table, plan)


def test_a_planned_decode_over_scattered_rows_gets_a_gather_instead():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache, rows=(0, 2))

    assert not cache.slot_table.is_window(plan.slot_mapping)
    with pytest.raises(SlotsUnsound, match="a prefix"):
        check_mapping_is_window(cache.slot_table, plan)


def test_a_planned_decode_writes_one_cell_per_row_per_step():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    for _ in range(5):
        plan_decode(cache)
    table = cache.slot_table

    assert table.appended_cells == 15
    assert table.resynced_cells == 6  # the prompts, taken once
    assert table.resyncs == 3
    check_appends_incremental(table, rows=3, steps=5)


def test_the_prompt_is_resynced_once_and_then_never_again():
    _, cache = _prefilled([[1, 2, 3], [4]])
    for _ in range(8):
        plan_decode(cache)

    assert cache.slot_table.resyncs == 2
    check_resyncs_bounded(cache.slot_table, rows=2)


def test_the_measured_host_work_is_a_fraction_of_what_day_50_wrote():
    """The same 8-step run Day 50 priced at 656 cells for 4 rows from 16 tokens."""
    _, cache = _prefilled([[1] * 4] * 4, num_blocks=64, block_size=4)
    for _ in range(8):
        plan_decode(cache)
    table = cache.slot_table
    written = table.appended_cells + table.resynced_cells

    assert written == 48  # 16 of prompt, 32 appended
    assert rebuild_cells(8, 4, 4) == 272
    assert written < rebuild_cells(8, 4, 4) / 5


def test_a_reset_row_is_resynced_when_its_next_tenant_decodes():
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan_decode(cache)
    before = cache.slot_table.resyncs
    cache.reset_row(1)
    assert cache.slot_table.length(1) == 0

    _prefill_rows(cache, cfg, [[7, 8]], (1,), seed=9)
    plan_decode(cache)

    assert cache.slot_table.resyncs == before + 1
    check_slots_agree(cache.slot_table, cache.tables, (0, 1))


def test_the_engine_sizes_its_table_from_max_model_len():
    model, _ = _model()
    engine = Engine.build(model, num_blocks=64, block_size=4, max_batch_size=2, max_model_len=32)

    assert engine.cache.max_model_len == 32
    assert tuple(engine.cache.slot_table.slots.shape) == (2, 32)


# --- a window is live storage -----------------------------------------------------


def test_a_later_step_cannot_change_a_held_windows_real_slots():
    """Appends write at column `length`, which is past the window's real region."""
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)
    real = [
        plan.slot_mapping[i, :n].tolist()
        for i, n in enumerate(plan.context_lens.tolist())
    ]
    plan_decode(cache)

    assert [
        plan.slot_mapping[i, :n].tolist()
        for i, n in enumerate(plan.context_lens.tolist())
    ] == real


def test_a_later_step_does_overwrite_a_short_rows_padding_inside_a_held_window():
    """And this is why the lengths cannot be a window too: the cell a later step
    writes is inside a rectangle somebody may still be holding, and the only thing
    keeping it out of the read is a `context_lens` that did not move."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    assert plan.slot_mapping[1, 2].item() == 0

    plan_decode(cache)

    assert plan.slot_mapping[1, 2].item() != 0
    assert plan.context_lens.tolist() == [4, 2]


def test_the_context_lengths_are_a_copy_and_not_a_window():
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)

    assert not cache.slot_table.is_window(plan.context_lens)
    lens = plan.context_lens.tolist()
    plan_decode(cache)
    assert plan.context_lens.tolist() == lens


def test_a_length_that_tracked_the_cache_would_disarm_the_staleness_gate():
    """Day 50's `check_plan_current` compares a plan's lengths against the tables.
    A `context_lens` that read live out of the table always agrees with them, so the
    gate stops being able to fail rather than starting to."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    plan_decode(cache)  # the cache moves on, so the first plan is stale

    with pytest.raises(PlanUnsound, match="stale"):
        check_plan_current(plan, cache)

    tracking = replace(plan, context_lens=torch.tensor([5, 3], dtype=torch.long))
    check_plan_current(tracking, cache)  # a live length: nothing to report, ever


def test_check_window_intact_catches_a_resync_under_a_held_window():
    """A held window over a row that has since changed tenant is a lie, and it is
    not one anybody raises on: the slots are legal and belong to somebody else."""
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    check_window_intact(plan)

    cache.reset_row(1)
    _prefill_rows(cache, cfg, [[7, 8]], (1,), seed=9)
    plan_decode(cache)

    with pytest.raises(SlotsUnsound, match="row 1"):
        check_window_intact(plan)


# --- the forward still agrees -------------------------------------------------------


def test_a_planned_decode_over_a_window_matches_one_over_a_rebuilt_rectangle():
    """The only claim that matters: this is a change of where the numbers live."""
    model, cfg = _model()
    prompts = [[1, 2, 3], [4], [5, 6]]
    _, cache = _prefilled(prompts)
    plan = plan_decode(cache)
    mapping, lens = rebuild_mapping(cache.tables, plan.rows)
    rebuilt = replace(plan, slot_mapping=mapping, context_lens=lens)

    ids = torch.tensor([[9], [9], [9]])
    windowed = model.forward(ids, plan.positions, cache=cache.view(plan.rows, plan=plan))
    replayed = model.forward(
        ids, rebuilt.positions, cache=cache.view(rebuilt.rows, plan=rebuilt)
    )

    assert torch.allclose(windowed, replayed, atol=1e-6)


def test_replaying_a_plan_still_returns_the_same_logits():
    """Day 50's headline property, over storage that other steps are writing into."""
    model, _ = _model()
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    ids = torch.tensor([[9], [9]])

    first = model.forward(ids, plan.positions, cache=cache.view(plan.rows, plan=plan))
    second = model.forward(ids, plan.positions, cache=cache.view(plan.rows, plan=plan))

    assert torch.allclose(first, second, atol=1e-6)


# --- what it costs -----------------------------------------------------------------


def test_the_table_is_rows_times_length_cells():
    assert table_cells(256, 2048) == 524288
    assert table_bytes(256, 2048) == 524288 * 8


def test_a_block_table_is_block_size_times_smaller():
    """What vLLM keeps. A slot is `block_id * block_size + offset`, so storing the
    block ids and computing the offset in the kernel is the same table divided by
    the block size, and it is why vLLM can afford one at 131k."""
    assert block_table_cells(256, 2048, 16) == 256 * 128
    assert block_table_bytes(256, 2048, 16) == 256 * 128 * 8
    assert table_bytes(256, 2048) / block_table_bytes(256, 2048, 16) == 16


def test_a_block_table_rounds_a_partial_block_up():
    assert block_table_cells(1, 17, 16) == 2


def test_the_incremental_cost_is_linear_in_the_steps():
    """One pass over the prompt, then one cell per row per step, forever."""
    assert incremental_cells(8, 4, 16) == 4 * 24
    assert incremental_cells(512, 4, 16) == 4 * 528
    assert incremental_cells(0, 4, 16) == 64


def test_the_incremental_cost_beats_the_rebuild_at_every_useful_length():
    for steps in (8, 64, 128, 512):
        assert incremental_cells(steps, 4, 16) < rebuild_cells(steps, 4, 16)
    assert rebuild_cells(512, 4, 16) / incremental_cells(512, 4, 16) > 200


def test_the_table_can_be_priced_against_the_pool_it_addresses():
    assert resident_share(table_bytes(8, 2048), 1 << 30) == pytest.approx(0.000122, abs=1e-5)


def test_the_arithmetic_refuses_shapes_that_are_not_shapes():
    with pytest.raises(ValueError, match="at least one row"):
        table_cells(0, 16)
    with pytest.raises(ValueError, match="at least one token"):
        table_cells(4, 0)
    with pytest.raises(ValueError, match="positive"):
        block_table_cells(4, 16, 0)
    with pytest.raises(ValueError, match="non-negative"):
        incremental_cells(-1, 4, 16)
    with pytest.raises(ValueError, match="positive"):
        resident_share(64, 0)


# --- gates ---------------------------------------------------------------------------


def test_check_slots_agree_passes_on_a_synced_table():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan_decode(cache)

    check_slots_agree(cache.slot_table, cache.tables, (0, 1, 2))


def test_check_slots_agree_catches_a_drifted_mirror():
    """The one failure mode a mirror has: two things that are supposed to say the
    same thing, and only one of them is right."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan_decode(cache)
    cache.slot_table.slots[0, 1] = 99

    with pytest.raises(SlotsUnsound, match="position 1"):
        check_slots_agree(cache.slot_table, cache.tables, (0, 1))


def test_check_slots_agree_catches_a_length_that_drifted():
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan_decode(cache)
    cache.slot_table.reset(1)

    with pytest.raises(SlotsUnsound, match="0 tokens"):
        check_slots_agree(cache.slot_table, cache.tables, (0, 1))


def test_check_mapping_is_window_refuses_a_rebuilt_rectangle():
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    mapping, lens = rebuild_mapping(cache.tables, plan.rows)
    rebuilt = replace(plan, slot_mapping=mapping, context_lens=lens)

    with pytest.raises(SlotsUnsound, match="its own allocation"):
        check_mapping_is_window(cache.slot_table, rebuilt)


def test_check_appends_incremental_fails_a_table_that_was_rebuilt():
    table = SlotTable(2, 32)
    for step in range(4):
        table.append((0, 1), [step * 2, step * 2 + 1])
    check_appends_incremental(table, rows=2, steps=4)

    with pytest.raises(SlotsUnsound, match="8 cells"):
        check_appends_incremental(table, rows=2, steps=8)


def test_check_resyncs_bounded_fails_a_table_that_keeps_rebuilding_a_row():
    tables = [_block_table(2)]
    table = SlotTable(1, 16)
    for _ in range(4):
        table.reset(0)
        table.sync(tables, (0,))

    with pytest.raises(SlotsUnsound, match="4 resyncs"):
        check_resyncs_bounded(table, rows=1)
    check_resyncs_bounded(table, rows=1, limit=4.0)


def test_the_resync_limit_leaves_room_for_a_preemption():
    """One resync a row is the prefill; the allowance is for the recompute a
    preempted request comes back through."""
    assert DEFAULT_RESYNC_LIMIT >= 2.0


def test_check_table_fits_refuses_a_table_that_eats_the_budget():
    check_table_fits(8, 2048, budget_bytes=1 << 20)

    with pytest.raises(SlotsUnsound, match="268435456 bytes"):
        check_table_fits(256, 131072, budget_bytes=1 << 26)


def test_check_table_fits_names_the_block_table_that_would_have_fit():
    with pytest.raises(SlotsUnsound, match="block ids"):
        check_table_fits(256, 131072, budget_bytes=1 << 26, block_size=16)
