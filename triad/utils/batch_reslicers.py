from concurrent.futures import ThreadPoolExecutor
import itertools
import math
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
import pyarrow as pa

from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size

from .iter import slice_iterable

T = TypeVar("T")
ArrowWorker = Callable[[Iterable[pa.Table]], pa.Table]


class BatchReslicer(Generic[T]):
    """Reslice batch streams with row or/and size limit

    :param row_limit: max row for each slice, defaults to None
    :param size_limit: max byte size for each slice, defaults to None

    :raises AssertionError: if `size_limit` is not None but `sizer` is None
    """

    def __init__(
        self,
        row_limit: Optional[int] = None,
        size_limit: Any = None,
    ) -> None:
        if row_limit is None:
            self._row_limit = 0
        else:
            self._row_limit = row_limit
        if size_limit is None:
            self._size_limit = 0
        else:
            self._size_limit = to_size(str(size_limit))

    def get_rows_and_size(self, batch: T) -> Tuple[int, int]:
        """Get the number of rows and byte size of a batch

        :param batch: the batch object
        :return: the number of rows and byte size of the batch
        """
        raise NotImplementedError  # pragma: no cover

    def take(self, batch: T, start: int, length: int) -> T:
        """Take a slice of the batch

        :param batch: the batch object
        :param start: the start row index
        :param length: the number of rows to take

        :return: a slice of the batch
        """
        raise NotImplementedError  # pragma: no cover

    def concat(self, batches: List[T]) -> T:
        """Concatenate a list of batches into one batch

        :param batches: the list of batches
        :return: the concatenated batch
        """
        raise NotImplementedError  # pragma: no cover

    def reslice(self, batches: Iterable[T]) -> Iterable[T]:  # noqa: C901, A003
        """Reslice the batch stream into new batches constrained by the row or size limit

        :param batches: the batch stream

        :yield: an iterable of batches of the same type with the constraints
        """
        if self._row_limit <= 0 and self._size_limit <= 0:
            for batch in batches:
                batch_rows, _ = self.get_rows_and_size(batch)
                if batch_rows > 0:
                    yield batch
            return
        cache: List[T] = []
        total_rows, total_size = 0, 0
        cache_rows = 0
        for batch in batches:
            batch_rows, batch_size = self.get_rows_and_size(batch)
            if batch_rows == 0:
                continue
            total_rows += batch_rows
            total_size += batch_size
            row_size = total_size / total_rows

            if self._row_limit > 0 and self._size_limit <= 0:
                row_limit = self._row_limit
            elif self._row_limit <= 0 and self._size_limit > 0:
                row_limit = max(math.floor(self._size_limit / row_size), 1)
            else:
                row_limit = min(
                    self._row_limit, max(math.floor(self._size_limit / row_size), 1)
                )

            if cache_rows >= row_limit:  # clean up edge cases
                yield self.concat(cache)
                cache = []
                cache_rows = 0

            if cache_rows + batch_rows < row_limit:
                cache.append(batch)
                cache_rows += batch_rows
            else:
                # here we guarantee initial_rows > 0
                slices, remain = self._slice_rows(
                    batch_rows, row_limit - cache_rows, slice_rows=row_limit
                )
                for i, rg in enumerate(slices):
                    chunk = self.take(batch, rg[0], rg[1])
                    if i == 0:
                        yield self.concat(cache + [chunk])
                        cache = []
                        cache_rows = 0
                    else:
                        yield chunk
                if remain[1] > 0:
                    cache.append(self.take(batch, remain[0], remain[1]))
                    cache_rows += remain[1]
        if len(cache) > 0:
            yield self.concat(cache)

    def _slice_rows(
        self, batch_rows: int, initial_rows: int, slice_rows: int
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
        start = 0
        if initial_rows >= batch_rows:
            return [(0, batch_rows)], (0, 0)
        slices = [(0, initial_rows)]
        start = initial_rows
        while True:
            if batch_rows - start < slice_rows:
                return slices, (start, batch_rows - start)
            slices.append((start, slice_rows))
            start += slice_rows


class PandasBatchReslicer(BatchReslicer[pd.DataFrame]):
    def get_rows_and_size(self, batch: pd.DataFrame) -> Tuple[int, int]:
        return batch.shape[0], batch.memory_usage(deep=True).sum()

    def take(self, batch: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        if start == 0 and length == batch.shape[0]:
            return batch
        return batch.iloc[start : start + length]

    def concat(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        if len(batches) == 1:
            return batches[0]
        return pd.concat(batches)


class ArrowTableBatchReslicer(BatchReslicer[pa.Table]):
    def get_rows_and_size(self, batch: pa.Table) -> Tuple[int, int]:
        return batch.num_rows, batch.nbytes

    def take(self, batch: pa.Table, start: int, length: int) -> pa.Table:
        if start == 0 and length == batch.num_rows:
            return batch
        return batch.slice(start, length)

    def concat(self, batches: List[pa.Table]) -> pa.Table:
        if len(batches) == 1:
            return batches[0]
        return pa.concat_tables(batches)


class NumpyArrayBatchReslicer(BatchReslicer[np.ndarray]):
    def get_rows_and_size(self, batch: np.ndarray) -> Tuple[int, int]:
        return batch.shape[0], batch.nbytes

    def take(self, batch: np.ndarray, start: int, length: int) -> np.ndarray:
        if start == 0 and length == batch.shape[0]:
            return batch
        return batch[start : start + length]

    def concat(self, batches: List[np.ndarray]) -> np.ndarray:
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches, axis=0)


class SortedBatchReslicer(Generic[T]):
    """Reslice batch streams (that are alredy sorted by keys) by keys.

    :param keys: group keys to reslice by
    """

    def __init__(
        self,
        keys: List[str],
    ) -> None:
        self._keys = keys
        self._last_row: Optional[np.ndarray] = None

    def take(self, batch: T, start: int, length: int) -> T:
        """Take a slice of the batch

        :param batch: the batch object
        :param start: the start row index
        :param length: the number of rows to take

        :return: a slice of the batch
        """
        raise NotImplementedError  # pragma: no cover

    def concat(self, batches: List[T]) -> T:
        """Concatenate a list of batches into one batch

        :param batches: the list of batches
        :return: the concatenated batch
        """
        raise NotImplementedError  # pragma: no cover

    def get_keys_ndarray(self, batch: T, keys: List[str]) -> np.ndarray:
        """Get the keys as a numpy array

        :param batch: the batch object
        :param keys: the keys to get

        :return: the keys as a numpy array
        """
        raise NotImplementedError  # pragma: no cover

    def get_batch_length(self, batch: T) -> int:
        """Get the number of rows in the batch

        :param batch: the batch object

        :return: the number of rows in the batch
        """
        raise NotImplementedError  # pragma: no cover

    def reslice(self, batches: Iterable[T]) -> Iterable[Iterable[T]]:  # noqa: C901, A003
        """Reslice the batch stream into a stream of iterable of batches of the
        same keys

        :param batches: the batch stream

        :yield: an iterable of iterable of batches containing same keys
        """

        def slicer(
            n: int, current: Tuple[bool, T], last: Optional[Tuple[bool, T]]
        ) -> bool:
            return current[0]

        def get_slices() -> Iterable[Tuple[bool, T]]:
            for batch in batches:
                if self.get_batch_length(batch) > 0:
                    yield from self._reslice_single(batch)

        def transform(data: Iterable[Tuple[bool, T]]) -> Iterable[T]:
            for _, batch in data:
                yield batch

        for res in slice_iterable(get_slices(), slicer):
            yield transform(res)

    def reslice_and_merge(self, batches: Iterable[T]) -> Iterable[T]:  # noqa: C901, A003
        """Reslice the batch stream into new batches, each containing the same keys

        :param batches: the batch stream

        :yield: an iterable of batches, each containing the same keys
        """

        cache: Optional[T] = None

        for batch in batches:
            if self.get_batch_length(batch) > 0:
                for diff, sub in self._reslice_single(batch):
                    if not diff:
                        cache = self.concat([cache, sub])  # type: ignore
                    else:
                        if cache is not None:
                            yield cache
                        cache = sub

        if cache is not None:
            yield cache

    def _reslice_single(self, batch: T) -> Iterable[Tuple[bool, T]]:
        a = self.get_keys_ndarray(batch, self._keys)
        b = np.roll(a, 1, axis=0)
        diff = self._diff(a, b)
        if self._last_row is not None:
            diff_from_last: bool = self._diff(a[0:1], self._last_row)[0]  # type: ignore
        else:
            diff_from_last = True
        self._last_row = a[-1:]
        points = np.where(diff)[0].tolist() + [a.shape[0]]
        if len(points) == 1:
            yield diff_from_last, batch
        else:
            for i in range(len(points) - 1):
                new_start = diff_from_last if i == 0 else True
                yield new_start, self.take(batch, points[i], points[i + 1] - points[i])

    def _diff(self, a: np.ndarray, b: np.ndarray) -> bool:
        return ((a == b) | ((a != a) & (b != b))).sum(axis=1) < len(self._keys)


class PandasSortedBatchReslicer(SortedBatchReslicer[pd.DataFrame]):
    def get_keys_ndarray(self, batch: pd.DataFrame, keys: List[str]) -> np.ndarray:
        return batch[keys].to_numpy()

    def get_batch_length(self, batch: pd.DataFrame) -> int:
        return batch.shape[0]

    def take(self, batch: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        return batch.iloc[start : start + length]

    def concat(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(batches)


class ArrowTableSortedBatchReslicer(SortedBatchReslicer[pa.Table]):
    def get_keys_ndarray(self, batch: pa.Table, keys: List[str]) -> np.ndarray:
        return batch.select(keys).to_pandas().to_numpy()

    def get_batch_length(self, batch: pa.Table) -> int:
        return batch.num_rows

    def take(self, batch: pa.Table, start: int, length: int) -> pa.Table:
        return batch.slice(start, length)

    def concat(self, batches: List[pa.Table]) -> pa.Table:
        return pa.concat_tables(batches)


def plan_contiguous_shards(
    group_row_counts: List[int], max_shards: int
) -> List[Tuple[int, int]]:
    """Split a contiguous sequence of groups into at most `max_shards`
    non-empty shards, minimizing the maximum number of rows of any shard.

    Groups are indivisible: a group (including all its duplicated rows) always
    stays inside one shard. This is the classic greedy number-partition problem
    with contiguous items, solved by binary searching the optimal per-shard row
    capacity, then splitting under-capacity shards until exactly
    ``min(max_shards, number of groups)`` non-empty shards are produced.

    :param group_row_counts: number of rows of each contiguous group
    :param max_shards: maximum number of non-empty shards (``K``), must be >= 1
    :return: a list of ``[group_start, group_end)`` intervals in group order,
        each non-empty and contiguous, with no group split across intervals

    :raises ValueError: if `max_shards` < 1 or any group has <= 0 rows
    """
    assert_or_throw(max_shards >= 1, ValueError("max_shards must be >= 1"))
    n = len(group_row_counts)
    if n == 0:
        return []
    assert_or_throw(
        all(c > 0 for c in group_row_counts),
        ValueError("every group must contain at least one row"),
    )
    if n == 1:
        return [(0, 1)]
    shard_count = min(max_shards, n)
    largest = max(group_row_counts)
    total = sum(group_row_counts)

    def fits(capacity: int) -> bool:
        used = 1
        loaded = 0
        for count in group_row_counts:
            if loaded + count > capacity:
                used += 1
                loaded = count
                if used > shard_count:
                    return False
            else:
                loaded += count
        return True

    low, high = largest, total
    while low < high:
        mid = (low + high) // 2
        if fits(mid):
            high = mid
        else:
            low = mid + 1
    capacity = low

    # greedy fill with the optimal capacity; every produced shard is <= capacity
    shards: List[List[int]] = []
    start = 0
    loaded = 0
    for i, count in enumerate(group_row_counts):
        if loaded + count > capacity:
            shards.append([start, i])
            start = i
            loaded = count
        else:
            loaded += count
    shards.append([start, n])

    # extra shards are obtained by splitting shards that hold > 1 group;
    # splitting never increases any shard's row count beyond `capacity`
    while len(shards) < shard_count:
        for j in range(len(shards)):
            start, end = shards[j]
            if end - start > 1:
                mid = (start + end) // 2
                shards[j : j + 1] = [[start, mid], [mid, end]]
                break
    return [(s, e) for s, e in shards]


class ArrowShardedParallelProcessor:
    """Run a worker over an ordered Arrow batch stream in parallel shards.

    The input batches must already be sorted by the declared business `keys`
    (integer or string columns, ``NULL`` keys are treated as one group). Rows
    of the same key form one group and a group may span multiple batches.
    Groups are never split: every row (including duplicated rows) is handed to
    exactly one worker call.

    The groups are divided into at most `max_shards` contiguous key intervals
    (:func:`~triad.utils.batch_reslicers.plan_contiguous_shards`) using the real
    input row counts as load, so the maximum shard row count is globally
    minimal. Each shard runs in its own worker through a thread pool of
    `max_shards` threads, and the real Arrow results are concatenated back in
    interval order. The output of every worker call must strictly match
    `output_schema` (field order, types and metadata included).

    A failure of any worker, or a schema mismatch, fails the whole processing:
    the remaining shards are cancelled and no partial result is returned.

    :param keys: ordered business key column names (non-empty)
    :param output_schema: the strictly required schema of every output table
    :param max_shards: maximum number of non-empty shards (``K``), must be >= 1
    :param input_schema: optional schema of the input; when provided, the keys
        and their integer/string types are validated up front, otherwise they
        are validated against the first non-empty input batch

    :raises ValueError: if `max_shards` < 1, `keys` is empty, a key is missing
        from the schema, or a key column is neither integer nor string
    """

    def __init__(
        self,
        keys: List[str],
        output_schema: pa.Schema,
        max_shards: int,
        input_schema: Optional[pa.Schema] = None,
    ) -> None:
        assert_or_throw(len(keys) > 0, ValueError("keys must be non-empty"))
        assert_or_throw(max_shards >= 1, ValueError("max_shards must be >= 1"))
        self._keys = list(keys)
        self._output_schema = output_schema
        self._max_shards = max_shards
        self._reslicer = ArrowTableSortedBatchReslicer(keys=self._keys)
        if input_schema is not None:
            self._check_schema(input_schema)

    def process(self, batches: Iterable[pa.Table], worker: ArrowWorker) -> pa.Table:
        """Process the sorted batch stream in parallel shards.

        :param batches: Arrow tables already sorted by the declared keys
        :param worker: callable taking the batches of one shard (an iterable of
            non-empty :external+pyarrow:class:`pyarrow.Table`) and returning a
            table strictly matching the declared output schema
        :return: the concatenation of the workers' real result tables in shard
            (key interval) order; an empty table with the declared schema for
            empty input

        :raises ValueError: if the input schema is missing a key column or has
            a non integer/string key column
        :raises Exception: if any worker raises, or any worker output does not
            strictly match the declared output schema
        """
        # materialize groups (zero-copy slices) so true row loads are known;
        # a group spanning batches stays together as one item
        iterator = iter(batches)
        first: Optional[pa.Table] = next((b for b in iterator if b.num_rows > 0), None)
        if first is None:
            return pa.Table.from_pylist([], schema=self._output_schema)
        # validate before the reslicer accesses the key columns
        self._check_schema(first.schema)
        stream: Iterable[pa.Table] = itertools.chain([first], iterator)
        groups: List[List[pa.Table]] = []
        group_counts: List[int] = []
        for group_batches in self._reslicer.reslice(stream):
            parts = [b for b in group_batches if b.num_rows > 0]
            if parts:
                groups.append(parts)
                group_counts.append(sum(b.num_rows for b in parts))

        intervals = plan_contiguous_shards(group_counts, self._max_shards)

        def run_shard(index: int) -> pa.Table:
            start, end = intervals[index]

            def shard_batches() -> Iterable[pa.Table]:
                for group in groups[start:end]:
                    yield from group

            try:
                result = worker(shard_batches())
            except Exception as e:
                raise RuntimeError(
                    f"worker failed on shard {index} (groups [{start}, {end})): {e}"
                ) from e
            if not isinstance(result, pa.Table):
                raise TypeError(
                    f"worker of shard {index} (groups [{start}, {end})) returned "
                    f"{type(result).__name__}, expected pyarrow.Table"
                )
            if not result.schema.equals(self._output_schema, check_metadata=True):
                raise ValueError(
                    f"worker of shard {index} (groups [{start}, {end})) returned "
                    f"schema {result.schema} which does not match the declared "
                    f"output schema {self._output_schema}"
                )
            return result

        with ThreadPoolExecutor(max_workers=self._max_shards) as pool:
            futures = [pool.submit(run_shard, i) for i in range(len(intervals))]
            results: List[Optional[pa.Table]] = []
            for i, future in enumerate(futures):
                try:
                    results.append(future.result())
                except BaseException:
                    # the whole processing fails: never return partial results
                    for pending in futures[i + 1 :]:
                        pending.cancel()
                    raise
        return pa.concat_tables(results)

    def _check_schema(self, schema: pa.Schema) -> None:
        for key in self._keys:
            assert_or_throw(
                key in schema.names,
                ValueError(f"key column '{key}' is not found in input schema"),
            )
            data_type = schema.field(key).type
            assert_or_throw(
                pa.types.is_integer(data_type)
                or pa.types.is_string(data_type)
                or pa.types.is_large_string(data_type),
                ValueError(
                    f"key column '{key}' must be integer or string, got {data_type}"
                ),
            )
