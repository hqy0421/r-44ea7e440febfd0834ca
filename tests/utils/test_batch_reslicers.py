from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from triad.utils.batch_reslicers import (
    ArrowTableBatchReslicer,
    ArrowShardedParallelProcessor,
    NumpyArrayBatchReslicer,
    PandasBatchReslicer,
    PandasSortedBatchReslicer,
    ArrowTableSortedBatchReslicer,
    plan_contiguous_shards,
)


class _MockSlicer(PandasBatchReslicer):
    def get_rows_and_size(self, batch: pd.DataFrame) -> Tuple[int, int]:
        return len(batch), len(batch) * 1024  # assume each row is 1KB


def test_base_batch_reslicer():
    def _test(chunks, row_limit, size_limit, expected):
        slicer = _MockSlicer(row_limit=row_limit, size_limit=size_limit)
        orig = []
        start = 0
        for x in chunks:
            orig.append(pd.DataFrame({"a": range(start, start + x)}))
            start += x
        dfs = list(slicer.reslice(orig))
        assert sum(chunks) == sum(len(t) for t in dfs)
        if len(dfs) > 0:
            assert pd.concat(dfs).values.tolist() == pd.concat(orig).values.tolist()
            assert list(len(t) for t in dfs) == expected

    _test([], None, None, [])
    _test([0], 0, 0, [])
    _test([0, 1, 0, 2], -1, 0, [1, 2])

    _test([], 2, 0, [])
    _test([0], 2, 0, [])
    _test([1], 1, 0, [1])
    _test([1], 2, 0, [1])
    _test([2], 2, 0, [2])
    _test([1, 1], 2, 0, [2])
    _test([2, 2], 1, 0, [1, 1, 1, 1])
    _test([2, 1], 2, 0, [2, 1])
    _test([1, 0, 2], 2, 0, [2, 1])
    _test([1, 2, 3], 2, 0, [2, 2, 2])
    _test([1, 2, 3], 10, 0, [6])
    _test([1, 2, 3], 1, 0, [1, 1, 1, 1, 1, 1])

    _test([], 0, "2k", [])
    _test([0], 0, "2k", [])
    _test([1], 0, 1024, [1])
    _test([1], 0, 1000, [1])
    _test([1], 0, 2048, [1])
    _test([2], 0, "2k", [2])
    _test([1, 1], 0, 1030, [1, 1])
    _test([2], 0, 1030, [1, 1])
    _test([1, 1], 0, "2k", [2])
    _test([2, 2], 0, "1k", [1, 1, 1, 1])
    _test([2, 1], 0, "2k", [2, 1])
    _test([1, 0, 2], 0, "2k", [2, 1])
    _test([1, 2, 3], 0, "2k", [2, 2, 2])
    _test([1, 2, 3], 0, "10m", [6])
    _test([1, 2, 3], 0, "1k", [1, 1, 1, 1, 1, 1])
    _test([1, 2, 3], 0, 1000, [1, 1, 1, 1, 1, 1])

    _test([1, 1], 3, 1030, [1, 1])
    _test([1, 2, 3], 1, "5m", [1, 1, 1, 1, 1, 1])


def test_pandas_reslicer():
    s = PandasBatchReslicer()
    df = pd.DataFrame({"a": range(10)})
    assert s.get_rows_and_size(df) == (10, df.memory_usage(deep=True).sum())
    assert s.take(df, 0, 10) is df
    assert s.take(df, 1, 5).values.tolist() == df.iloc[1:6].values.tolist()
    assert s.concat([df]) is df
    assert s.concat([df, df]).values.tolist() == pd.concat([df, df]).values.tolist()

    # edge case: the adjusted row limit is smaller or equal to cache rows
    dfs1 = [pd.DataFrame({"a": ["abc"] * 100})] * 1
    dfs2 = [pd.DataFrame({"a": ["abc" * 100] * 10})] * 10
    dfs = dfs1 + dfs2 + dfs1
    s = PandasBatchReslicer(size_limit=dfs1[0].memory_usage(deep=True).sum() + 200)
    chunks = list(s.reslice(dfs))
    assert all(len(x) > 0 for x in chunks)
    assert pd.concat(chunks).values.tolist() == pd.concat(dfs).values.tolist()


def test_arrow_reslicer():
    s = ArrowTableBatchReslicer()
    pdf = pd.DataFrame({"a": range(10)})
    df = pa.Table.from_pandas(pdf)
    assert s.get_rows_and_size(df) == (10, df.nbytes)
    assert s.take(df, 0, 10) is df
    assert s.take(df, 1, 5).to_pandas().values.tolist() == pdf.iloc[1:6].values.tolist()
    assert s.concat([df]) is df
    assert (
        s.concat([df, df]).to_pandas().values.tolist()
        == pd.concat([pdf, pdf]).values.tolist()
    )


def test_numpy_reslicer():
    s = NumpyArrayBatchReslicer()
    pdf = pd.DataFrame({"a": range(10)})
    df = pdf.to_numpy()
    assert s.get_rows_and_size(df) == (10, df.nbytes)
    assert s.take(df, 0, 10) is df
    assert (s.take(df, 1, 5) == np.arange(1, 6)[:, None]).all()
    assert s.concat([df]) is df
    assert (s.concat([df, df]) == pd.concat([pdf, pdf]).to_numpy()).all()


def test_sorted_pandas_reslicer():
    def _make_df(data):
        for x in data:
            yield pd.DataFrame(
                {
                    "a": x,
                    "b": pd.Series([None] * len(x), dtype="object"),
                    "c": range(len(x)),
                }
            )

    def _test(data, expected, mode="first"):
        s = PandasSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for dfs in s.reslice(_make_df(data)):
            if mode == "first":
                df = next(dfs)
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                dfs = list(dfs)
                actual.append(str(dfs[0].iloc[0]["a"]) + "-" + str(len(pd.concat(dfs))))
        assert actual == expected
        s = PandasSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for df in s.reslice_and_merge(_make_df(data)):
            if mode == "first":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(len(df)))
        assert actual == expected

    _test([], [])
    _test([[]], [])
    _test([[1, 1]], ["1-0"])
    _test([[1, 1, 2]], ["1-0", "2-2"])

    _test([[1, 1], [], [1, 1], []], ["1-0"])
    _test([[1, 1], [], [1, 2], []], ["1-0", "2-1"])
    _test([[1, 1], [], [2, 2], []], ["1-0", "2-0"])
    _test([[], [1, 2], [], [2, 2], []], ["1-0", "2-1"])
    _test([[], [1], [1], [2], [2]], ["1-0", "2-0"])

    _test([[], [1], [1, 1, 2], [], [2], [], [2]], ["1-3", "2-3"], mode="all")


def test_sorted_arrow_reslicer():
    def _make_df(data):
        for x in data:
            yield pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "a": x,
                        "b": pd.Series([None] * len(x), dtype="object"),
                        "c": range(len(x)),
                    }
                )
            )

    def _test(data, expected, mode="first"):
        s = ArrowTableSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for dfs in s.reslice(_make_df(data)):
            if mode == "first":
                df = next(dfs).to_pandas()
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                dfs = [x.to_pandas() for x in dfs]
                actual.append(str(dfs[0].iloc[0]["a"]) + "-" + str(len(pd.concat(dfs))))
        assert actual == expected
        s = ArrowTableSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for df in s.reslice_and_merge(_make_df(data)):
            df = df.to_pandas()
            if mode == "first":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(len(df)))
        assert actual == expected

    _test([], [])
    _test([[]], [])
    _test([[1, 1]], ["1-0"])
    _test([[1, 1, 2]], ["1-0", "2-2"])

    _test([[1, 1], [], [1, 1], []], ["1-0"])
    _test([[1, 1], [], [1, 2], []], ["1-0", "2-1"])
    _test([[1, 1], [], [2, 2], []], ["1-0", "2-0"])
    _test([[], [1, 2], [], [2, 2], []], ["1-0", "2-1"])
    _test([[], [1], [1], [2], [2]], ["1-0", "2-0"])

    _test([[], [1], [1, 1, 2], [], [2], [], [2]], ["1-3", "2-3"], mode="all")


def test_plan_contiguous_shards_basic():
    assert plan_contiguous_shards([], 3) == []
    assert plan_contiguous_shards([5], 3) == [(0, 1)]
    # at most K non-empty shards, fewer when groups run out
    assert plan_contiguous_shards([2, 2, 2], 10) == [(0, 1), (1, 2), (2, 3)]
    assert plan_contiguous_shards([1, 1, 1, 1, 1], 3) == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    # the single big group can never be split
    assert plan_contiguous_shards([4, 1, 1, 1, 4], 2) == [(0, 3), (3, 5)]
    assert plan_contiguous_shards([4, 1, 1, 1, 4], 3) == [
        (0, 1),
        (1, 4),
        (4, 5),
    ]


def test_plan_contiguous_shards_validation():
    with pytest.raises(ValueError):
        plan_contiguous_shards([1, 2], 0)
    with pytest.raises(ValueError):
        plan_contiguous_shards([1, 0], 2)


def _dp_optimal(weights: List[int], k: int) -> int:
    n = len(weights)
    k = min(k, n)
    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for kk in range(1, k + 1):
        for i in range(kk, n + 1):
            dp[kk][i] = min(
                max(dp[kk - 1][j], prefix[i] - prefix[j]) for j in range(kk - 1, i)
            )
    return dp[k][n]


def test_plan_contiguous_shards_globally_optimal():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        weights = [int(x) for x in rng.integers(1, 15, size=n)]
        k = int(rng.integers(1, n + 4))
        intervals = plan_contiguous_shards(weights, k)
        # exactly min(K, n) non-empty contiguous shards
        assert len(intervals) == min(k, n)
        assert intervals[0][0] == 0
        assert intervals[-1][1] == n
        assert all(
            intervals[i][1] == intervals[i + 1][0] for i in range(len(intervals) - 1)
        )
        assert all(end > start for start, end in intervals)
        # no group lost and no group duplicated
        assert sum(e - s for s, e in intervals) == n
        max_load = max(sum(weights[s:e]) for s, e in intervals)
        assert max_load == _dp_optimal(weights, k)
        # no shard can hold fewer rows than the biggest single group
        assert max_load >= max(weights)


def _int_table(keys_with_nulls, extra=None):
    if extra is None:
        extra = list(range(len(keys_with_nulls)))
    return pa.table(
        {
            "a": pa.array(keys_with_nulls, type=pa.int64()),
            "c": pa.array(extra, type=pa.int64()),
        }
    )


def _split_batches(table, sizes):
    start = 0
    for size in sizes:
        yield table.slice(start, size)
        start += size


def test_arrow_sharded_processor_balancing_and_completeness():
    # groups (null, 1, 2, 3, 5, 8) with row counts [3, 4, 1, 2, 5, 2]
    spec = [(None, 3), (1, 4), (2, 1), (3, 2), (5, 5), (8, 2)]
    keys: List = []
    for key, count in spec:
        keys += [key] * count
    table = _int_table(keys)
    batches = list(_split_batches(table, [0, 2, 5, 0, 6, 4]))
    out_schema = pa.schema([pa.field("a", pa.int64()), pa.field("rows", pa.int64())])

    seen = []

    def worker(shard_batches):
        combined = pa.concat_tables(list(shard_batches))
        seen.append(
            (
                list(dict.fromkeys(combined.column("a").to_pylist())),
                combined.num_rows,
                combined.column("c").to_pylist(),
            )
        )
        return pa.Table.from_pylist(
            [{"a": combined.column("a")[0].as_py(), "rows": combined.num_rows}],
            schema=out_schema,
        )

    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=out_schema, max_shards=3
    )
    result = processor.process(batches, worker)

    # globally optimal loads [7, 3, 7], max load 7
    assert [load for _, load, _ in seen] == [7, 3, 7]
    # groups stay intact and stay in key-interval order, null first
    assert [group for groups, _, _ in seen for group in groups] == [
        None,
        1,
        2,
        3,
        5,
        8,
    ]
    # every input row (incl. duplicates) is delivered exactly once
    delivered = sorted(value for _, _, values in seen for value in values)
    assert delivered == list(range(17))
    # real arrow output, merged in shard order, strict schema
    assert result.schema.equals(out_schema, check_metadata=True)
    assert result.column("rows").to_pylist() == [7, 3, 7]


def test_arrow_sharded_processor_group_never_split_across_batches():
    # one key repeated over many tiny batches must land in a single shard
    table = _int_table([1, 1, 1, 2, 2])
    batches = list(_split_batches(table, [1, 1, 1, 1, 1]))
    out_schema = pa.schema([pa.field("a", pa.int64())])
    seen = []

    def worker(shard_batches):
        combined = pa.concat_tables(list(shard_batches))
        seen.append((combined.column("a").to_pylist()[0], combined.num_rows))
        return combined.select(["a"]).cast(out_schema)

    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=out_schema, max_shards=4
    )
    result = processor.process(batches, worker)
    # only 2 groups -> at most 2 non-empty shards, each group wholly inside one
    assert sorted(seen) == [(1, 3), (2, 2)]
    assert result.num_rows == 5
    assert result.column("a").to_pylist() == [1, 1, 1, 2, 2]


def test_arrow_sharded_processor_null_string_and_composite_keys():
    out_schema = pa.schema([pa.field("k", pa.string()), pa.field("n", pa.int64())])
    # group row counts [2, 3, 1, 4], the NULL key is its own group
    spec = [("a", 2), ("b", 3), (None, 1), ("d", 4)]
    keys: List = []
    for key, count in spec:
        keys += [key] * count
    table = pa.table({"k": pa.array(keys, type=pa.string()), "v": range(10)})

    def worker(shard_batches):
        # emit one row per group the shard received, preserving group order
        combined = pa.concat_tables(list(shard_batches))
        rows = []
        last_key, group_size = None, 0
        for key in combined.column("k").to_pylist():
            if (
                group_size > 0
                and key != last_key
                and not (key is None and last_key is None)
            ):
                rows.append({"k": last_key, "n": group_size})
                group_size = 0
            last_key = key
            group_size += 1
        if group_size:
            rows.append({"k": last_key, "n": group_size})
        return pa.Table.from_pylist(rows, schema=out_schema)

    processor = ArrowShardedParallelProcessor(
        keys=["k"], output_schema=out_schema, max_shards=2
    )
    result = processor.process([table.slice(0, 3), table.slice(3, 7)], worker)
    # optimal split of [2, 3, 1, 4] into 2 shards is [a, b] | [NULL, d],
    # groups stay intact and NULL is preserved as a real group
    assert result.column("k").to_pylist() == ["a", "b", None, "d"]
    assert result.column("n").to_pylist() == [2, 3, 1, 4]

    # composite keys, NULL sub-key rows stay in the same group
    rows = [(1, None), (1, None), (1, "x"), (2, "x"), (2, "x"), (3, None)]
    composite = pa.table(
        {
            "a": pa.array([r[0] for r in rows], type=pa.int64()),
            "b": pa.array([r[1] for r in rows], type=pa.string()),
        }
    )
    comp_schema = composite.schema
    loads = []

    def comp_worker(shard_batches):
        combined = pa.concat_tables(list(shard_batches))
        loads.append(combined.num_rows)
        return combined.cast(comp_schema)

    processor = ArrowShardedParallelProcessor(
        keys=["a", "b"], output_schema=comp_schema, max_shards=2
    )
    result = processor.process(
        [composite.slice(0, 2), composite.slice(2, 2), composite.slice(4, 2)],
        comp_worker,
    )
    assert result.num_rows == 6
    assert sum(loads) == 6


def test_arrow_sharded_processor_empty_input_typed_result():
    out_schema = pa.schema([pa.field("a", pa.int64())])
    called = []

    def worker(shard_batches):  # pragma: no cover - must never be called
        called.append(True)
        return pa.Table.from_pylist([{"a": 1}], schema=out_schema)

    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=out_schema, max_shards=3
    )
    for empty in ([], [_int_table([])]):
        result = processor.process(empty, worker)
        assert result.num_rows == 0
        assert result.schema.equals(out_schema, check_metadata=True)
    assert called == []


def test_arrow_sharded_processor_worker_failure_fails_all():
    table = _int_table(list(range(9)))
    out_schema = pa.schema([pa.field("a", pa.int64())])

    def worker(shard_batches):
        combined = pa.concat_tables(list(shard_batches))
        if combined.num_rows == 3:
            raise ValueError("kaboom")
        return combined.select(["a"]).cast(out_schema)

    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=out_schema, max_shards=3
    )
    with pytest.raises(RuntimeError, match="worker failed") as exc_info:
        processor.process([table], worker)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_arrow_sharded_processor_schema_mismatch_fails_all():
    table = _int_table([1, 2, 3])
    good_schema = pa.schema([pa.field("a", pa.int64())])

    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=good_schema, max_shards=1
    )
    # wrong type
    with pytest.raises(ValueError, match="does not match the declared output"):
        processor.process(
            [table],
            lambda b: pa.concat_tables(list(b))
            .select(["a"])
            .cast(pa.schema([pa.field("a", pa.int32())])),
        )
    # wrong field order
    two_col = pa.table(
        {
            "a": pa.array([1], type=pa.int64()),
            "b": pa.array([2], type=pa.int64()),
        }
    )
    processor = ArrowShardedParallelProcessor(
        keys=["a"],
        output_schema=pa.schema([pa.field("b", pa.int64()), pa.field("a", pa.int64())]),
        max_shards=1,
    )
    with pytest.raises(ValueError):
        processor.process([two_col], lambda b: pa.concat_tables(list(b)))
    # metadata difference is also a strict mismatch
    meta_schema = pa.schema([pa.field("a", pa.int64(), metadata={"x": "1"})])
    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=meta_schema, max_shards=1
    )
    with pytest.raises(ValueError):
        processor.process([table], lambda b: pa.concat_tables(list(b)).select(["a"]))
    # non-table return
    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=good_schema, max_shards=1
    )
    with pytest.raises(TypeError):
        processor.process([table], lambda b: 42)


def test_arrow_sharded_processor_input_validation():
    out_schema = pa.schema([pa.field("a", pa.int64())])
    with pytest.raises(ValueError):
        ArrowShardedParallelProcessor(keys=[], output_schema=out_schema, max_shards=1)
    with pytest.raises(ValueError):
        ArrowShardedParallelProcessor(
            keys=["a"], output_schema=out_schema, max_shards=0
        )
    # bad key type declared up front
    with pytest.raises(ValueError, match="integer or string"):
        ArrowShardedParallelProcessor(
            keys=["a"],
            output_schema=out_schema,
            max_shards=1,
            input_schema=pa.schema([pa.field("a", pa.bool_())]),
        )
    # missing key at construction time
    with pytest.raises(ValueError, match="not found"):
        ArrowShardedParallelProcessor(
            keys=["missing"],
            output_schema=out_schema,
            max_shards=1,
            input_schema=out_schema,
        )
    # missing key at runtime surfaces as ValueError rather than KeyError
    processor = ArrowShardedParallelProcessor(
        keys=["missing"], output_schema=out_schema, max_shards=1
    )
    with pytest.raises(ValueError, match="not found"):
        processor.process([_int_table([1])], lambda b: b)
    # bad key type at runtime
    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=out_schema, max_shards=1
    )
    with pytest.raises(ValueError, match="integer or string"):
        processor.process(
            [pa.table({"a": pa.array([True, False], type=pa.bool_())})],
            lambda b: b,
        )


def test_arrow_sharded_processor_runs_shards_in_parallel():
    import time

    table = _int_table(list(range(9)))
    out_schema = pa.schema([pa.field("a", pa.int64())])

    def slow_worker(shard_batches):
        list(shard_batches)
        time.sleep(0.3)
        return pa.Table.from_pylist([{"a": 1}], schema=out_schema)

    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=out_schema, max_shards=3
    )
    start = time.time()
    processor.process([table], slow_worker)
    # 3 shards x 0.3s would take ~0.9s serially; parallel must be much less
    assert time.time() - start < 0.7


def test_arrow_sharded_processor_reusable_and_single_shard():
    table = _int_table([1, 1, 2, 3, 3, 3])
    out_schema = table.schema
    processor = ArrowShardedParallelProcessor(
        keys=["a"], output_schema=out_schema, max_shards=1
    )

    def identity(shard_batches):
        return pa.concat_tables(list(shard_batches))

    first = processor.process([table.slice(0, 3), table.slice(3, 3)], identity)
    second = processor.process([table], identity)  # instance reused
    assert first.column("a").to_pylist() == table.column("a").to_pylist()
    assert second.column("a").to_pylist() == table.column("a").to_pylist()
