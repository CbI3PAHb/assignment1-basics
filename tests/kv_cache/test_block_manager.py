"""CPU admission credits and lazy physical pages for the serving scheduler.

Public API: generation.kv_cache.block_manager.BlockManager(
    *, num_blocks: int, block_size: int
)
reserve(*, request_id, max_tokens) -> bool reserves rounded block credits, not pages.
True creates a new reservation; False refuses insufficient credits (including
capacity larger than the pool) or a duplicate ID, without changing any state.
ensure_capacity(*, request_id, num_tokens) -> tuple[int, ...] grows its block table.
get_block_table(*, request_id) -> tuple[int, ...]; release(*, request_id) is idempotent.
free_block_count counts physical blocks; available_block_count counts credits.
The allocator permits the last reserved page's slack; exact token limits belong
to the scheduler. Required total blocks must not exceed the request's credits.
Physical block selection/order is deliberately not prescribed by these tests.
"""

import importlib

import pytest


@pytest.fixture
def manager_type():
    module_name = "cs336_basics.generation.kv_cache.block_manager"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            pytest.fail(
                "Implement BlockManager in generation/kv_cache/block_manager.py; "
                "the CPU contract is documented at the top of this test file.",
                pytrace=False,
            )
        raise
    return module.BlockManager


def assert_table(table, *, size, num_blocks):
    assert isinstance(table, tuple), "A plan needs an immutable block-table snapshot"
    assert len(table) == size
    assert len(set(table)) == size
    assert all(isinstance(block, int) and 0 <= block < num_blocks for block in table)


def test_reservation_consumes_admission_credits_without_assigning_pages(manager_type):
    manager = manager_type(num_blocks=5, block_size=4)
    assert manager.free_block_count == manager.available_block_count == 5

    assert manager.reserve(request_id="A", max_tokens=9) is True  # ceil(9 / 4) = 3
    assert manager.available_block_count == 2
    assert manager.free_block_count == 5
    assert manager.get_block_table(request_id="A") == ()
    assert manager.ensure_capacity(request_id="A", num_tokens=0) == ()
    assert manager.free_block_count == 5


def test_growth_keeps_old_addresses_and_allocates_only_across_page_boundaries(
    manager_type,
):
    manager = manager_type(num_blocks=4, block_size=4)
    assert manager.reserve(request_id="A", max_tokens=10)

    first = manager.ensure_capacity(request_id="A", num_tokens=3)
    assert_table(first, size=1, num_blocks=4)
    assert manager.free_block_count == 3
    assert manager.ensure_capacity(request_id="A", num_tokens=4) == first

    second = manager.ensure_capacity(request_id="A", num_tokens=5)
    assert_table(second, size=2, num_blocks=4)
    assert second[:1] == first
    assert manager.get_block_table(request_id="A") == second
    assert manager.free_block_count == 2

    third = manager.ensure_capacity(request_id="A", num_tokens=10)
    assert_table(third, size=3, num_blocks=4)
    assert third[:2] == second
    assert first == third[:1], "Previously issued snapshots must not grow in place"
    assert manager.ensure_capacity(request_id="A", num_tokens=2) == third  # never shrink
    assert manager.free_block_count == manager.available_block_count == 1


def test_reserved_requests_can_grow_even_when_no_admission_credits_remain(
    manager_type,
):
    manager = manager_type(num_blocks=4, block_size=4)
    assert manager.reserve(request_id="A", max_tokens=8)
    assert manager.reserve(request_id="B", max_tokens=8)
    a_first = manager.ensure_capacity(request_id="A", num_tokens=1)
    b_first = manager.ensure_capacity(request_id="B", num_tokens=1)
    assert manager.available_block_count == 0
    assert manager.free_block_count == 2

    # Physically free pages are already promised to A/B; C must wait.
    assert manager.reserve(request_id="C", max_tokens=1) is False
    assert manager.available_block_count == 0
    assert manager.free_block_count == 2
    with pytest.raises(KeyError):
        manager.get_block_table(request_id="C")

    a_full = manager.ensure_capacity(request_id="A", num_tokens=8)
    b_full = manager.ensure_capacity(request_id="B", num_tokens=8)
    assert_table(a_full, size=2, num_blocks=4)
    assert_table(b_full, size=2, num_blocks=4)
    assert a_full[:1] == a_first
    assert b_full[:1] == b_first
    assert set(a_full).isdisjoint(b_full)
    assert manager.free_block_count == 0

    manager.release(request_id="A")
    assert manager.free_block_count == manager.available_block_count == 2
    assert manager.reserve(request_id="C", max_tokens=8)
    c_full = manager.ensure_capacity(request_id="C", num_tokens=8)
    assert set(c_full) == set(a_full), "Only A's released pages are available"
    assert manager.get_block_table(request_id="B") == b_full
    assert set(c_full).isdisjoint(b_full)


def test_release_returns_unused_credits_and_actual_pages_exactly_once(manager_type):
    manager = manager_type(num_blocks=5, block_size=4)
    assert manager.reserve(request_id="A", max_tokens=12)
    assert manager.reserve(request_id="B", max_tokens=4)
    b_table = manager.ensure_capacity(request_id="B", num_tokens=1)
    manager.ensure_capacity(request_id="A", num_tokens=1)

    manager.release(request_id="A")
    assert manager.free_block_count == manager.available_block_count == 4
    assert manager.get_block_table(request_id="B") == b_table
    with pytest.raises(KeyError):
        manager.get_block_table(request_id="A")
    manager.release(request_id="A")
    manager.release(request_id="never-reserved")
    assert manager.free_block_count == manager.available_block_count == 4

    manager.release(request_id="B")
    assert manager.free_block_count == manager.available_block_count == 5
    assert manager.reserve(request_id="A", max_tokens=20)
    assert manager.get_block_table(request_id="A") == ()


@pytest.mark.parametrize("num_tokens", [-1, 9])
def test_growth_respects_reserved_block_capacity_without_partial_mutation(
    manager_type,
    num_tokens,
):
    manager = manager_type(num_blocks=3, block_size=4)
    assert manager.reserve(request_id="A", max_tokens=5)
    first = manager.ensure_capacity(request_id="A", num_tokens=3)
    assert_table(first, size=1, num_blocks=3)

    # Five tokens reserve two blocks: all eight slots are within that capacity.
    before = manager.ensure_capacity(request_id="A", num_tokens=6)
    assert_table(before, size=2, num_blocks=3)
    assert before[:1] == first
    assert manager.ensure_capacity(request_id="A", num_tokens=8) == before
    assert manager.free_block_count == manager.available_block_count == 1

    # Nine needs a third block despite only ONE block being missing.
    with pytest.raises(ValueError):
        manager.ensure_capacity(request_id="A", num_tokens=num_tokens)
    assert manager.get_block_table(request_id="A") == before
    assert manager.free_block_count == 1
    assert manager.available_block_count == 1


def test_impossible_and_duplicate_reservations_leave_existing_ownership_intact(
    manager_type,
):
    manager = manager_type(num_blocks=3, block_size=4)
    assert manager.reserve(request_id="A", max_tokens=4)
    before = manager.ensure_capacity(request_id="A", num_tokens=1)
    assert manager.reserve(request_id="too-large", max_tokens=13) is False
    assert manager.get_block_table(request_id="A") == before
    assert manager.free_block_count == manager.available_block_count == 2
    with pytest.raises(KeyError):
        manager.get_block_table(request_id="too-large")

    assert manager.reserve(request_id="A", max_tokens=8) is False
    with pytest.raises(KeyError):
        manager.ensure_capacity(request_id="unknown", num_tokens=1)
    assert manager.get_block_table(request_id="A") == before
    assert manager.free_block_count == manager.available_block_count == 2
