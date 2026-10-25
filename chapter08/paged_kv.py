"""A teaching model of a paged KV cache: fixed-size blocks, per-request block tables, reference-counted sharing,
copy-on-write, and hash-based prefix caching. It manages block indices only (no tensors), so it runs anywhere and
shows the bookkeeping that vLLM's block manager does.

    python paged_kv.py
"""
from collections import OrderedDict


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.free = list(range(num_blocks))[::-1]          # physical block ids
        self.refcount = [0] * num_blocks
        self.tables = {}                                   # request id -> list of physical blocks
        self.lengths = {}                                  # request id -> tokens stored
        self.prefix = OrderedDict()                        # block hash -> physical block (LRU order)

    # -- allocation -------------------------------------------------------------------------------------------
    def _alloc(self) -> int:
        if not self.free:                                  # reclaim a cached prefix block nobody uses
            for h, b in list(self.prefix.items()):
                if self.refcount[b] == 0:
                    del self.prefix[h]; self.free.append(b); break
        if not self.free:
            raise MemoryError("KV cache full: the scheduler must preempt a request")
        b = self.free.pop(); self.refcount[b] = 1
        return b

    def add_request(self, rid, prompt_tokens):
        """Allocate blocks for a prompt, reusing cached full blocks with the same prefix."""
        table, h, hits = [], None, 0
        bs = self.block_size
        for start in range(0, len(prompt_tokens), bs):
            chunk = tuple(prompt_tokens[start:start + bs])
            h = hash((h, chunk))                           # a block's hash depends on its whole prefix
            if len(chunk) == bs and h in self.prefix:
                b = self.prefix[h]; self.prefix.move_to_end(h); self.refcount[b] += 1; hits += 1
            else:
                b = self._alloc()
                if len(chunk) == bs:
                    # simplification: a real engine publishes the block only after its K and V are computed
                    self.prefix[h] = b                     # full blocks become reusable prefixes
            table.append(b)
        self.tables[rid], self.lengths[rid] = table, len(prompt_tokens)
        return hits                                        # blocks whose prefill can be skipped

    def append_token(self, rid):
        """Decode step: room for one more token, allocating or copying on write when needed."""
        table, n = self.tables[rid], self.lengths[rid]
        if n % self.block_size == 0:
            table.append(self._alloc())
        elif self.refcount[table[-1]] > 1:                 # last block is shared: copy before writing
            self.refcount[table[-1]] -= 1
            table[-1] = self._alloc()                      # (a real engine also copies the block's K and V)
        self.lengths[rid] = n + 1

    def fork(self, rid, new_rid):
        """Parallel sampling or beam search: share every block, copy lazily on the first write."""
        for b in self.tables[rid]:
            self.refcount[b] += 1
        self.tables[new_rid], self.lengths[new_rid] = list(self.tables[rid]), self.lengths[rid]

    def free_request(self, rid):
        for b in self.tables.pop(rid):
            self.refcount[b] -= 1
            if self.refcount[b] == 0 and b not in self.prefix.values():
                self.free.append(b)                        # cached prefix blocks stay until evicted
        self.lengths.pop(rid)

    def used_blocks(self):
        return sum(1 for r in self.refcount if r > 0)


if __name__ == "__main__":
    kv = PagedKVCache(num_blocks=64, block_size=16)
    system = list(range(1000, 1048))                       # a 48-token system prompt: three full blocks
    print("request A, prefix hits:", kv.add_request("A", system + [1, 2, 3, 4, 5]))
    print("request B, prefix hits:", kv.add_request("B", system + [9, 9, 9]))
    for _ in range(20):
        kv.append_token("A")
    kv.fork("B", "B2")                                     # a second sample of B shares all of B's blocks
    kv.append_token("B2")                                  # copy-on-write of B2's last block
    print("tables:", {r: t for r, t in kv.tables.items()})
    print("blocks in use:", kv.used_blocks(), "of 64")
