"""Hand-written data structures and algorithms used by the booking desk.

Nothing here touches the database, so every structure can be unit-tested and
explained on its own. (Python's `heapq`, `collections.deque`, `dict` and
`sorted` are deliberately NOT used for these.)

  MaxHeap          priority queue            -> priority booking requests
  Deque            doubly-linked FIFO queue  -> flight waitlists
  Stack            LIFO                      -> undo last booking
  BST              binary search tree        -> flight search by flight number
  HashTable        chained hash table        -> passenger / ticket lookup
  SinglyLinkedList chain of nodes            -> flight manifest (detach cancelled bookings)
  quicksort        in-place quicksort        -> sorting search results
"""
from __future__ import annotations

from typing import Any, Callable, Iterator


# ===================================================================== heap ==
class MaxHeap:
    """Array-based binary max-heap. The parent of index i is (i-1)//2 and its
    children are 2i+1 and 2i+2. Push and pop are O(log n); peek is O(1).

    Entries are (priority, item); priorities are compared with `>` so any
    comparable value works (we use tuples such as (tier_rank, -arrival_id))."""

    def __init__(self) -> None:
        self._a: list[tuple[Any, Any]] = []

    def __len__(self) -> int:
        return len(self._a)

    def push(self, priority: Any, item: Any) -> None:
        a = self._a
        a.append((priority, item))
        i = len(a) - 1
        while i > 0:                                   # sift up
            parent = (i - 1) // 2
            if a[i][0] > a[parent][0]:
                a[i], a[parent] = a[parent], a[i]
                i = parent
            else:
                break

    def peek(self) -> Any:
        if not self._a:
            raise IndexError("peek from empty heap")
        return self._a[0][1]

    def pop(self) -> Any:
        a = self._a
        if not a:
            raise IndexError("pop from empty heap")
        top = a[0][1]
        last = a.pop()
        if a:
            a[0] = last
            i, n = 0, len(a)
            while True:                                # sift down
                left, right, big = 2 * i + 1, 2 * i + 2, i
                if left < n and a[left][0] > a[big][0]:
                    big = left
                if right < n and a[right][0] > a[big][0]:
                    big = right
                if big == i:
                    break
                a[i], a[big] = a[big], a[i]
                i = big
        return top

    def array(self) -> list:
        """Items in heap-array (level) order, for showing the internal layout."""
        return [item for _, item in self._a]

    def in_priority_order(self) -> list:
        """Non-destructive: the order pop() would return, highest first."""
        clone = MaxHeap()
        clone._a = list(self._a)
        return [clone.pop() for _ in range(len(clone))]


# ==================================================================== deque ==
class _DNode:
    __slots__ = ("value", "prev", "next")

    def __init__(self, value: Any) -> None:
        self.value, self.prev, self.next = value, None, None


class Deque:
    """Double-ended queue built from a doubly-linked list with head and tail
    pointers. All four end operations and removing a known node are O(1)."""

    def __init__(self) -> None:
        self.head: _DNode | None = None
        self.tail: _DNode | None = None
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def push_back(self, value: Any) -> _DNode:
        node = _DNode(value)
        if self.tail is None:
            self.head = self.tail = node
        else:
            node.prev, self.tail.next = self.tail, node
            self.tail = node
        self._n += 1
        return node

    def push_front(self, value: Any) -> _DNode:
        node = _DNode(value)
        if self.head is None:
            self.head = self.tail = node
        else:
            node.next, self.head.prev = self.head, node
            self.head = node
        self._n += 1
        return node

    def peek_front(self) -> Any:
        if self.head is None:
            raise IndexError("peek from empty deque")
        return self.head.value

    def pop_front(self) -> Any:
        if self.head is None:
            raise IndexError("pop from empty deque")
        node = self.head
        self.remove(node)
        return node.value

    def pop_back(self) -> Any:
        if self.tail is None:
            raise IndexError("pop from empty deque")
        node = self.tail
        self.remove(node)
        return node.value

    def remove(self, node: _DNode) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        node.prev = node.next = None
        self._n -= 1

    def __iter__(self) -> Iterator[Any]:
        node = self.head
        while node:
            yield node.value
            node = node.next

    def to_list(self) -> list:
        return list(self)


# ==================================================================== stack ==
class Stack:
    """Last-in, first-out. push, pop and peek are O(1) (list tail)."""

    def __init__(self) -> None:
        self._a: list = []

    def __len__(self) -> int:
        return len(self._a)

    def push(self, item: Any) -> None:
        self._a.append(item)

    def pop(self) -> Any:
        if not self._a:
            raise IndexError("pop from empty stack")
        return self._a.pop()

    def peek(self) -> Any:
        if not self._a:
            raise IndexError("peek from empty stack")
        return self._a[-1]

    def top_first(self, limit: int | None = None) -> list:
        items = self._a[::-1]
        return items if limit is None else items[:limit]


# ====================================================================== BST ==
class _BNode:
    __slots__ = ("key", "value", "left", "right")

    def __init__(self, key: Any, value: Any) -> None:
        self.key, self.value, self.left, self.right = key, value, None, None


class BST:
    """Binary search tree: left subtree < node < right subtree.

    Search and insert are O(h) where h is the height: O(log n) when balanced,
    O(n) when keys arrive already sorted. `build_balanced` therefore inserts a
    sorted list median-first, which keeps the height near log2(n).
    All loops are iterative, so a deep tree cannot hit Python's recursion limit."""

    def __init__(self) -> None:
        self.root: _BNode | None = None
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def insert(self, key: Any, value: Any) -> None:
        if self.root is None:
            self.root, self._n = _BNode(key, value), 1
            return
        node = self.root
        while True:
            if key == node.key:
                node.value = value
                return
            side = "left" if key < node.key else "right"
            nxt = getattr(node, side)
            if nxt is None:
                setattr(node, side, _BNode(key, value))
                self._n += 1
                return
            node = nxt

    def search(self, key: Any) -> dict:
        """Returns value (or None), the number of key comparisons made, the
        path of keys visited, and for a miss the nearest smaller/larger keys."""
        node, comparisons, path = self.root, 0, []
        below = above = None
        while node:
            path.append(node.key)
            comparisons += 1
            if key == node.key:
                return {"found": True, "value": node.value, "comparisons": comparisons,
                        "path": path, "below": None, "above": None}
            if key < node.key:
                above, node = node.key, node.left
            else:
                below, node = node.key, node.right
        return {"found": False, "value": None, "comparisons": comparisons,
                "path": path, "below": below, "above": above}

    def height(self) -> int:
        if not self.root:
            return 0
        best, level = 0, [self.root]
        while level:
            best += 1
            level = [c for n in level for c in (n.left, n.right) if c]
        return best

    def inorder(self) -> Iterator[tuple[Any, Any]]:
        stack, node = [], self.root
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key, node.value
            node = node.right

    @classmethod
    def build_balanced(cls, sorted_pairs: list[tuple[Any, Any]]) -> "BST":
        tree = cls()
        todo = [(0, len(sorted_pairs) - 1)]
        while todo:                                    # insert the middle first
            lo, hi = todo.pop()
            if lo > hi:
                continue
            mid = (lo + hi) // 2
            tree.insert(*sorted_pairs[mid])
            todo.append((lo, mid - 1))
            todo.append((mid + 1, hi))
        return tree


# ============================================================== hash table ==
class HashTable:
    """Hash table with separate chaining. A key is hashed to a bucket index;
    keys that collide share a short list. Average put/get/remove is O(1); the
    table doubles when the load factor passes 0.75 to keep chains short."""

    def __init__(self, capacity: int = 64) -> None:
        self._buckets: list[list[list]] = [[] for _ in range(capacity)]
        self._n = 0

    def __len__(self) -> int:
        return self._n

    @staticmethod
    def _hash(key: str) -> int:
        h = 5381                                       # djb2
        for ch in key:
            h = ((h * 33) ^ ord(ch)) & 0xFFFFFFFF
        return h

    def _index(self, key: str) -> int:
        return self._hash(key) % len(self._buckets)

    def put(self, key: str, value: Any) -> None:
        bucket = self._buckets[self._index(key)]
        for pair in bucket:
            if pair[0] == key:
                pair[1] = value
                return
        bucket.append([key, value])
        self._n += 1
        if self._n / len(self._buckets) > 0.75:
            self._grow()

    def get(self, key: str) -> dict:
        idx = self._index(key)
        comparisons = 0
        for k, v in self._buckets[idx]:
            comparisons += 1
            if k == key:
                return {"found": True, "value": v, "bucket": idx, "comparisons": comparisons,
                        "chain": len(self._buckets[idx])}
        return {"found": False, "value": None, "bucket": idx, "comparisons": comparisons,
                "chain": len(self._buckets[idx])}

    def remove(self, key: str) -> bool:
        bucket = self._buckets[self._index(key)]
        for i, pair in enumerate(bucket):
            if pair[0] == key:
                bucket.pop(i)
                self._n -= 1
                return True
        return False

    def _grow(self) -> None:
        old = self._buckets
        self._buckets = [[] for _ in range(len(old) * 2)]
        for bucket in old:
            for k, v in bucket:
                self._buckets[self._index(k)].append([k, v])

    def stats(self) -> dict:
        return {"entries": self._n, "buckets": len(self._buckets),
                "load_factor": round(self._n / len(self._buckets), 2),
                "longest_chain": max((len(b) for b in self._buckets), default=0)}


# ======================================================== singly linked list ==
class _SNode:
    __slots__ = ("value", "next")

    def __init__(self, value: Any) -> None:
        self.value, self.next = value, None


class SinglyLinkedList:
    """Chain of nodes, each pointing to the next. Appending is O(1) (tail
    pointer). Detaching a node is O(n) to find it, then O(1) to unlink: the
    previous node simply points past it, so the rest of the chain stays intact."""

    def __init__(self) -> None:
        self.head: _SNode | None = None
        self.tail: _SNode | None = None
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def append(self, value: Any) -> None:
        node = _SNode(value)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self._n += 1

    def detach(self, match: Callable[[Any], bool]) -> dict | None:
        """Unlink the first node whose value satisfies `match`. Returns the
        removed value plus its former neighbours, or None if absent."""
        prev, node = None, self.head
        while node and not match(node.value):
            prev, node = node, node.next
        if node is None:
            return None
        if prev is None:
            self.head = node.next
        else:
            prev.next = node.next
        if node is self.tail:
            self.tail = prev
        self._n -= 1
        return {"value": node.value,
                "before": prev.value if prev else None,
                "after": node.next.value if node.next else None}

    def __iter__(self) -> Iterator[Any]:
        node = self.head
        while node:
            yield node.value
            node = node.next

    def to_list(self) -> list:
        return list(self)


# =================================================================== quicksort ==
def quicksort(items: list, key: Callable[[Any], Any] = lambda x: x) -> tuple[list, int]:
    """Sort ascending with quicksort; returns (new sorted list, comparisons).

    * median-of-three pivot, so already-sorted input is not the worst case
    * Lomuto partition, iterative (explicit stack) so no recursion-depth limit
    * ranges of 10 or fewer finish with insertion sort
    * each key is paired with the item's original position, which makes the
      sort stable: equal keys keep their original order
    Average O(n log n), worst case O(n^2) (rare with median-of-three)."""
    a = [(key(x), i, x) for i, x in enumerate(items)]
    comparisons = 0

    def less(p, q) -> bool:
        nonlocal comparisons
        comparisons += 1
        return (p[0], p[1]) < (q[0], q[1])

    todo = [(0, len(a) - 1)]
    while todo:
        lo, hi = todo.pop()
        if hi - lo < 10:                               # small range: insertion sort
            for i in range(lo + 1, hi + 1):
                cur, j = a[i], i - 1
                while j >= lo and less(cur, a[j]):
                    a[j + 1] = a[j]
                    j -= 1
                a[j + 1] = cur
            continue
        mid = (lo + hi) // 2
        if less(a[mid], a[lo]):
            a[mid], a[lo] = a[lo], a[mid]
        if less(a[hi], a[lo]):
            a[hi], a[lo] = a[lo], a[hi]
        if less(a[hi], a[mid]):
            a[hi], a[mid] = a[mid], a[hi]
        a[mid], a[hi - 1] = a[hi - 1], a[mid]          # median now at hi-1
        pivot, store = a[hi - 1], lo + 1
        for i in range(lo + 1, hi - 1):
            if less(a[i], pivot):
                a[i], a[store] = a[store], a[i]
                store += 1
        a[store], a[hi - 1] = a[hi - 1], a[store]
        todo.append((lo, store - 1))
        todo.append((store + 1, hi))
    return [x for _, _, x in a], comparisons
