"""Unit tests for the hand-written data structures (no database needed)."""
import random

import pytest

from app.dsa import BST, Deque, HashTable, MaxHeap, SinglyLinkedList, Stack, quicksort


def test_max_heap_pops_in_priority_order_with_ties_by_arrival():
    h = MaxHeap()
    # priority = (tier_rank, -arrival_id): higher tier first, earlier arrival first on ties
    for arrival, tier in enumerate([1, 3, 2, 3, 1, 2], start=1):
        h.push((tier, -arrival), f"req{arrival}")
    assert h.peek() == "req2"
    assert h.in_priority_order() == ["req2", "req4", "req3", "req6", "req1", "req5"]
    assert len(h) == 6                                   # in_priority_order is non-destructive
    assert [h.pop() for _ in range(6)] == ["req2", "req4", "req3", "req6", "req1", "req5"]
    with pytest.raises(IndexError):
        h.pop()


def test_max_heap_matches_sorted_on_random_data():
    rng = random.Random(1)
    vals = [rng.randint(0, 10_000) for _ in range(500)]
    h = MaxHeap()
    for i, v in enumerate(vals):
        h.push((v, i), v)
    assert [h.pop() for _ in vals] == sorted(vals, reverse=True)


def test_deque_is_fifo_and_supports_both_ends_and_middle_removal():
    d = Deque()
    nodes = [d.push_back(x) for x in "abcd"]
    assert d.to_list() == list("abcd") and len(d) == 4
    assert d.pop_front() == "a"                           # FIFO
    d.push_front("z")
    assert d.to_list() == list("zbcd")
    d.remove(nodes[2])                                    # remove "c" from the middle in O(1)
    assert d.to_list() == list("zbd")
    assert d.pop_back() == "d" and d.peek_front() == "z"
    while len(d):
        d.pop_front()
    assert d.head is None and d.tail is None
    with pytest.raises(IndexError):
        d.pop_front()


def test_stack_is_lifo():
    s = Stack()
    for x in (1, 2, 3):
        s.push(x)
    assert s.peek() == 3 and s.top_first() == [3, 2, 1] and s.top_first(2) == [3, 2]
    assert [s.pop(), s.pop(), s.pop()] == [3, 2, 1]
    with pytest.raises(IndexError):
        s.pop()


def test_bst_search_and_neighbours():
    t = BST()
    for k in [50, 30, 70, 20, 40, 60, 80]:
        t.insert(k, f"v{k}")
    r = t.search(60)
    assert r["found"] and r["value"] == "v60" and r["path"] == [50, 70, 60] and r["comparisons"] == 3
    miss = t.search(65)
    assert not miss["found"] and miss["below"] == 60 and miss["above"] == 70
    assert [k for k, _ in t.inorder()] == [20, 30, 40, 50, 60, 70, 80]


def test_bst_balanced_build_stays_shallow_even_for_sorted_input():
    pairs = [(f"SB{n:05d}", n) for n in range(4000)]       # already sorted: worst case for naive insert
    t = BST.build_balanced(pairs)
    assert len(t) == 4000 and t.height() <= 13              # about log2(4000) = 12
    for n in (0, 1999, 3999):
        r = t.search(f"SB{n:05d}")
        assert r["found"] and r["value"] == n and r["comparisons"] <= 13
    naive = BST()
    for k, v in pairs[:300]:
        naive.insert(k, v)
    assert naive.height() == 300                             # degenerates into a linked list


def test_hash_table_put_get_update_remove_and_grow():
    h = HashTable(capacity=4)
    for i in range(1000):
        h.put(f"KEY-{i}", i)
    assert len(h) == 1000 and h.stats()["buckets"] >= 1000 / 0.75 / 2
    assert h.stats()["load_factor"] <= 0.75
    r = h.get("KEY-777")
    assert r["found"] and r["value"] == 777 and r["comparisons"] >= 1
    h.put("KEY-777", "new")
    assert h.get("KEY-777")["value"] == "new" and len(h) == 1000
    assert h.remove("KEY-777") and not h.get("KEY-777")["found"] and not h.remove("KEY-777")
    assert len(h) == 999


def test_linked_list_detach_keeps_chain_intact():
    ll = SinglyLinkedList()
    for x in "ABCDE":
        ll.append(x)
    out = ll.detach(lambda v: v == "C")                      # middle
    assert out == {"value": "C", "before": "B", "after": "D"} and ll.to_list() == list("ABDE")
    assert ll.detach(lambda v: v == "A")["before"] is None and ll.to_list() == list("BDE")  # head
    assert ll.detach(lambda v: v == "E")["after"] is None and ll.to_list() == list("BD")    # tail
    ll.append("F")                                           # tail pointer still correct
    assert ll.to_list() == list("BDF") and len(ll) == 3
    assert ll.detach(lambda v: v == "Q") is None


@pytest.mark.parametrize("seed", range(5))
def test_quicksort_matches_sorted_and_is_stable(seed):
    rng = random.Random(seed)
    data = [(rng.randint(0, 50), i) for i in range(400)]     # many duplicate keys
    out, comps = quicksort(data, key=lambda t: t[0])
    assert out == sorted(data, key=lambda t: t[0])           # Python's sort is stable too
    assert comps > 0


def test_quicksort_edge_cases():
    assert quicksort([])[0] == [] and quicksort([5])[0] == [5]
    asc = list(range(2000))
    assert quicksort(asc)[0] == asc and quicksort(asc[::-1])[0] == asc
    assert quicksort([3] * 100)[0] == [3] * 100
