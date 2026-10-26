import attrs

from cs336_basics.config._validation import (
    _NONNEGATIVE_INTEGER,
    _POSITIVE_INTEGER,
)


def ceil(a: int, b: int) -> int:
    """Divide an integer count by a block size, rounding up.

    Args:
        a: Nonnegative item count, such as a token count.
        b: Positive number of items per block.

    Returns:
        int: Number of blocks needed; zero when a is zero.

    Raises:
        ValueError: If a < 0 or b <= 0

    Notes:
        Validate the count and block size before performing the division.
    """
    if a < 0:
        raise ValueError(f"got {a=} < 0")
    if b <= 0:
        raise ValueError(f"got {b=} < 0")

    return (a + b - 1) // b


@attrs.define(kw_only=True, eq=False)
class BlockManager:
    """Manage CPU reservation credits and physical KV block ownership.

    Reserving credits promises future space without assigning block IDs.
    Growing a request assigns physical blocks from that promise. Releasing a
    request returns both its physical blocks and all its reserved credits.
    GPU tensor allocation and exact token/context limits belong elsewhere.

    Args:
        num_blocks: Total number of physical blocks; a positive integer.
        block_size: Token slots per block; a positive integer.
        block_tables: Manager-owned request-to-block-ID lists, empty by default.
            Use the default when constructing a new manager.
        reserved_block_counts: Manager-owned request-to-credit counts, empty by
            default. Use the default when constructing a new manager.

    Attributes:
        available_block_count: Unreserved credits, equal to num_blocks minus
            the sum of all reserved_block_counts.
        free_block_count: Physically unassigned blocks, equal to the size of
            free_block_pool. Reserved requests may still need these blocks.
        free_block_pool: Unassigned physical IDs from 0 through num_blocks - 1.

    Raises:
        TypeError: If a constructor argument fails its attrs type validator.
        ValueError: If num_blocks or block_size is nonpositive or a boolean.

    Notes:
        Construct with empty ownership mappings; supplying populated mappings
        does not restore counters or a free pool. Do not mutate state directly.
    """

    # Total physical capacity of the pool.

    num_blocks: int = attrs.field(validator=_POSITIVE_INTEGER)
    block_size: int = attrs.field(validator=_POSITIVE_INTEGER)

    # request_id to block_table
    block_tables: dict[str, list[int]] = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    reserved_block_counts: dict[str, int] = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))

    available_block_count: int = attrs.field(init=False, validator=_NONNEGATIVE_INTEGER)
    free_block_count: int = attrs.field(init=False, validator=_NONNEGATIVE_INTEGER)
    free_block_pool: set[int] = attrs.field(init=False, validator=attrs.validators.instance_of(set))

    def __attrs_post_init__(self) -> None:
        """Initialize counters and free physical IDs for a new manager.

        attrs calls this hook during construction. Both counters start at
        num_blocks, and all IDs in range(num_blocks) start free.

        Returns:
            None.

        Notes:
            No additional arguments or explicit exceptions. Constructor field
            validation is described in the class docstring. This is not a reset
            operation for a manager that already owns reservations.
        """
        self.available_block_count = self.num_blocks
        self.free_block_count = self.num_blocks
        self.free_block_pool = set(range(self.num_blocks))

    def update_free_block_count(self):
        self.free_block_count = len(self.free_block_pool)

    def get_block_table(self, *, request_id: str) -> tuple[int, ...]:
        """Return an immutable snapshot of a request's physical block IDs.

        Args:
            request_id: ID of a currently reserved request.

        Returns:
            tuple[int, ...]: Physical IDs ordered by logical block position.
            Empty if reserved but no pages have been assigned. Later growth
            does not change previously returned tuples.

        Raises:
            KeyError: If request_id was never reserved, its reservation was
                refused, or it has already been released.

        Notes:
            An empty table for a reserved request is valid and does not raise.
            This method never allocates blocks or changes state.
        """
        if request_id not in self.block_tables:
            raise KeyError(f"block table for request_id='{request_id}' doesn't exists")

        return tuple(self.block_tables[request_id])

    def ensure_capacity(self, *, request_id: str, num_tokens: int) -> tuple[int, ...]:
        """Assign enough physical blocks for a reserved request's total length.

        Append only the missing block IDs. Preserve the existing table prefix,
        never shrink the table, and leave reservation credits unchanged.

        Args:
            request_id: ID of a currently reserved request.
            num_tokens: Total token capacity required, not the number of new
                tokens. Must be nonnegative; zero is allowed.

        Returns:
            tuple[int, ...]: Snapshot of the complete block table after growth.
            If the existing table is already large enough, return its current
            contents without allocating or freeing blocks.

        Raises:
            ValueError: If num_tokens is negative, or ceil(num_tokens /
                block_size) exceeds this request's reserved_block_counts entry.
                Compare TOTAL required blocks, not just missing blocks.
            KeyError: If request_id has no current reservation.

        Notes:
            Validate before changing tables, the free pool, or counters.
            Zero tokens, repeated calls, and a smaller requested length are
            valid and do not raise. Zero available admission credits also does
            not prevent growth within an existing reservation: its pages have
            already been promised.

            The last reserved block's unused slots are allowed: reserving five
            tokens with block_size=4 permits capacity for six or eight tokens;
            nine requires a third block and raises ValueError.
        """
        if num_tokens < 0:
            raise ValueError(f"got {num_tokens=} < 0")

        if request_id not in self.block_tables:
            raise KeyError(f"request_id={request_id} not in block tables")

        required_blocks = ceil(num_tokens, self.block_size)
        if required_blocks > self.reserved_block_counts[request_id]:
            raise ValueError(
                f"Trying to allocate allocate more blocks, then were reserved"
                f" got {required_blocks=}, {self.reserved_block_counts[request_id]}"
            )

        if required_blocks <= len(self.block_tables[request_id]):
            return tuple(self.block_tables[request_id])

        missing_blocks = required_blocks - len(self.block_tables[request_id])
        assert missing_blocks > 0, f"{missing_blocks=}"

        for _ in range(missing_blocks):
            block_id = self.free_block_pool.pop()
            self.block_tables[request_id].append(block_id)
        self.update_free_block_count()

        return tuple(self.block_tables[request_id])

    def reserve(self, *, request_id: str, max_tokens: int) -> bool:
        """Try to reserve block credits for a new request without assigning pages.

        A successful call reserves ceil(max_tokens / block_size) credits and
        creates an empty block table. It decreases available_block_count but
        leaves free_block_count and free_block_pool unchanged.

        Args:
            request_id: Stable ID for the reservation to create.
            max_tokens: Positive total token capacity to reserve, including
                prompt and future tokens, not just newly generated tokens.
                Rounded up to whole blocks; not an exact per-token limit.

        Returns:
            bool: True if a NEW reservation was created. False if request_id
            already has a reservation, or the required block count exceeds
            available_block_count. This includes requests larger than the whole
            pool. False leaves ALL state unchanged, including an existing table.

        Raises:
            ValueError: If max_tokens is zero or negative.

        Notes:
            For positive max_tokens, duplicate IDs and insufficient capacity
            return False; neither condition raises ValueError or KeyError.
            Never reset, extend, or replace an existing reservation here.
            Invalid max_tokens must also leave state unchanged.
        """
        if max_tokens < 0:
            raise ValueError(f"got {max_tokens=}")

        if request_id in self.block_tables:
            return False

        required_blocks = ceil(max_tokens, self.block_size)
        if required_blocks > self.available_block_count:
            return False

        self.available_block_count -= required_blocks
        self.block_tables[request_id] = []
        self.reserved_block_counts[request_id] = required_blocks
        return True

    def release(self, *, request_id: str) -> bool:
        """Release a request's physical blocks and all of its reservation credits.

        Return assigned block IDs to the free pool, restore all reserved credits
        (including unused ones), and remove both ownership entries. This frees
        block ownership; it does not clear or deallocate GPU tensor storage.

        Args:
            request_id: ID to release; it may be unknown or already released.

        Returns:
            bool: True if an existing reservation was removed, even if it had
            no assigned pages. False if there was no reservation; state remains
            unchanged. Repeating release does not return credits/pages twice.

        Raises:
            No expected exceptions for valid manager state. Unknown or already
            released IDs return False, not KeyError or ValueError.
        """
        if request_id not in self.block_tables:
            return False

        block_table = self.get_block_table(request_id=request_id)

        for block_id in block_table:
            self.free_block_pool.add(block_id)

        self.available_block_count += self.reserved_block_counts[request_id]
        self.free_block_count += len(block_table)
        del self.reserved_block_counts[request_id]
        del self.block_tables[request_id]
        return True
