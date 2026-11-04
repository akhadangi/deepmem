from deepmem.data import iter_byte_batches, iter_documents


def test_local_jsonl_and_partition(tmp_path):
    f = tmp_path / 'sample.jsonl'
    f.write_text(''.join(f'{{"text":"doc{i:03d}"}}\n' for i in range(20)), encoding='utf-8')
    a = list(iter_documents(data_files=str(f), seed=7, rank=0, world_size=2))
    b = list(iter_documents(data_files=str(f), seed=7, rank=1, world_size=2))
    assert len(a) == len(b) == 10
    assert set(a).isdisjoint(b)
    assert len(set(a + b)) == 20


def test_local_byte_batches(tmp_path):
    f = tmp_path / 'sample.txt'
    f.write_text('A' * 80 + '\n', encoding='utf-8')
    it = iter_byte_batches(data_files=str(f), dataset_name=None, dataset_config=None,
                           split='train', text_column='text', block_size=32,
                           batch_size=1, seed=1, repeat=False)
    batches = list(it)
    assert len(batches) == 2
    assert batches[0].shape == (1, 32)
    assert batches[0].max().item() < 256


def test_shard_one_long_document_by_block(tmp_path):
    f = tmp_path / "long.txt"
    f.write_text("a" * 32 * 16, encoding="utf-8")
    from deepmem.data import iter_byte_batches
    counts = []
    for rank in range(4):
        batches = list(iter_byte_batches(data_files=str(f), dataset_name=None,
            dataset_config=None, split="train", text_column="text", block_size=32,
            batch_size=1, seed=42, rank=rank, world_size=4, repeat=False))
        counts.append(len(batches))
    assert counts == [4, 4, 4, 4]
