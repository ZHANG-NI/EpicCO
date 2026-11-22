#!/usr/bin/env python3
"""make_bpp2d_seedset.py <shapes.json> [seed]: the fixed training set of the 2D guillotine bin packing island group,
written as the training directory of a pseudo-campaign `bpp_seed` so that setup_camp.py can copy it
(train_copy_from=bpp_seed).

The instances are NEW draws (they are not benchmark files). Four families, after the four public families:
  c  identical bins. The ten classes of the literature: classes 1-6 = item sides uniform on [1,10], [1,10], [1,35],
     [1,35], [1,100], [1,100] in square bins of side 10, 30, 40, 100, 100, 300; classes 7-10 = bin 100 x 100 and four
     item types (wide-flat, narrow-tall, large, small), one of them drawn with probability 0.7, the others 0.1 each.
  v  variable bins, unlimited stock: the items of a class as above; bin types = the class bin and four more with both
     sides uniform on [side/2, side] (duplicates dropped).
  g  variable bins, limited stock, many items per bin: 2-6 bin types with sides uniform on [150, 1000]; items = the
     pieces of random guillotine cuts of rectangles that fit the largest bin type; "nice" pieces (cuts near the middle of the
     longer side) or "path" pieces (cuts anywhere: extreme aspect ratios); stock = 3 to 6 times the item area.
  h  six small bin types (sides 10-60 x 10-120), limited stock of 1.6 to 2 times the item area, item sides 1-30 with
     aspect ratio at most 7, about half as many item types as items.
shapes.json = [[family, n items, class or kind], ...]; file names carry no family label (T00-<k>-n<items>)."""
import json
import random
import sys
from pathlib import Path

SE = Path(__file__).resolve().parent.parent
shapes = json.loads(Path(sys.argv[1]).read_text())
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261114
out = SE / "camp" / "bpp_seed" / "priv" / "train"
if out.exists():
    sys.exit("bpp_seed already exists: not overwritten")
rng = random.Random(seed)

SIDE = {1: 10, 2: 30, 3: 40, 4: 100, 5: 100, 6: 300, 7: 100, 8: 100, 9: 100, 10: 100}
HI = {1: 10, 2: 10, 3: 35, 4: 35, 5: 100, 6: 100}


def class_items(c, n):
    W = H = SIDE[c]
    its = []
    for _ in range(n):
        if c <= 6:
            its.append((rng.randint(1, HI[c]), rng.randint(1, HI[c])))
        else:
            main = c - 6
            t = main if rng.random() < 0.7 else rng.choice([x for x in (1, 2, 3, 4) if x != main])
            if t == 1:
                its.append((rng.randint((2 * W + 2) // 3, W), rng.randint(1, H // 2)))
            elif t == 2:
                its.append((rng.randint(1, W // 2), rng.randint((2 * H + 2) // 3, H)))
            elif t == 3:
                its.append((rng.randint(W // 2, W), rng.randint(H // 2, H)))
            else:
                its.append((rng.randint(1, W // 2), rng.randint(1, H // 2)))
    return W, H, its


def cut_pieces(rects, n, nice):
    """Cut the rectangles into n pieces in total by random guillotine cuts (integer sides >= 1)."""
    pcs = list(rects)
    guard = 0
    while len(pcs) < n and guard < 100000:
        guard += 1
        if nice:                                   # always the largest piece, near the middle of its longer side
            k = max(range(len(pcs)), key=lambda i: pcs[i][0] * pcs[i][1])
        else:                                      # a random piece, weighted by area
            tot = sum(a * b for a, b in pcs)
            x, k, acc = rng.random() * tot, 0, 0.0
            for i, (a, b) in enumerate(pcs):
                acc += a * b
                if acc >= x:
                    k = i
                    break
        a, b = pcs[k]
        along_a = (a >= b) if nice else (rng.random() < 0.5)
        side = a if along_a else b
        if side < 2:
            along_a = not along_a
            side = a if along_a else b
            if side < 2:
                continue
        lo, hi = (0.3, 0.7) if nice else (0.03, 0.97)
        p = min(side - 1, max(1, int(round(side * rng.uniform(lo, hi)))))
        pcs[k] = (p, b) if along_a else (a, p)
        pcs.append((side - p, b) if along_a else (a, side - p))
    return pcs


def fam_g(n, nice):
    k = rng.randint(2, 6)
    bins = []
    while len(bins) < k:
        b = (rng.randint(150, 1000), rng.randint(150, 1000))
        if b not in bins and b[0] != b[1]:
            bins.append(b)
    bl, bh = max(bins, key=lambda x: x[0] * x[1])            # the largest bin type: every source rectangle fits it
    big = bl * bh
    target = rng.uniform(1.4, 5.0) * big
    rects, area = [], 0
    while area < target and len(rects) < max(1, n // 5):
        r = (rng.randint(max(2, bl // 2), bl), rng.randint(max(2, bh // 2), bh))
        rects.append(r)
        area += r[0] * r[1]
    its = cut_pieces(rects, n, nice)
    ia = sum(a * b for a, b in its)
    ratio = rng.uniform(3.0, 6.0)
    stock = [1] * k
    while sum(s * a * b for s, (a, b) in zip(stock, bins)) < ratio * ia:
        j = rng.randrange(k)
        if stock[j] < 9 or all(s >= 9 for s in stock):
            stock[j] += 1
    return [(a, b, s) for (a, b), s in zip(bins, stock)], its


def fam_h(n):
    bins = []
    while len(bins) < 6:
        b = (rng.randint(10, 60), rng.randint(10, 120))
        if b not in bins:
            bins.append(b)
    types = []
    while len(types) < max(2, n // 2):
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        if max(a, b) <= 7 * min(a, b) and any(a <= L and b <= H for L, H in bins):
            types.append((a, b))
    its = list(types) + [rng.choice(types) for _ in range(n - len(types))]
    ia = sum(a * b for a, b in its)
    ratio = rng.uniform(1.6, 2.0)
    stock = [2] * 6
    while sum(s * a * b for s, (a, b) in zip(stock, bins)) < ratio * ia:
        stock[rng.randrange(6)] += 1
    return [(a, b, s) for (a, b), s in zip(bins, stock)], its


def write(name, bins, its):
    cnt = {}
    for it in its:
        cnt[it] = cnt.get(it, 0) + 1
    order = list(cnt)
    rng.shuffle(order)
    items = [{"Length": a, "Height": b, "Demand": cnt[(a, b)], "DemandMax": None, "Value": a * b} for a, b in order]
    objs = [{"Length": a, "Height": b, "Stock": s, "Cost": a * b} for a, b, s in bins]
    for a, b in order:
        assert any(a <= L and b <= H for L, H, _ in bins), "an item that fits no bin"
    (out / "inst" / (name + ".json")).write_text(json.dumps({"Name": name, "Objects": objs, "Items": items}, indent=0))


(out / "inst").mkdir(parents=True)
names = []
for k, (fam, n, arg) in enumerate(shapes, 1):
    name = "T00-%02d-n%d" % (k, n)
    if fam == "c":
        W, H, its = class_items(int(arg), n)
        bins = [(W, H, None)]
    elif fam == "v":
        W, H, its = class_items(int(arg), n)
        b = [(W, H)]
        for _ in range(4):
            x = (rng.randint((W + 1) // 2, W), rng.randint((H + 1) // 2, H))
            if x not in b:
                b.append(x)
        bins = [(a, c, None) for a, c in b]
    elif fam == "g":
        bins, its = fam_g(n, arg == "nice")
    elif fam == "h":
        bins, its = fam_h(n)
    else:
        sys.exit("unknown family %r" % (fam,))
    assert len(its) == n, (name, len(its), n)
    write(name, bins, its)
    names.append(name)
(out / "initial.json").write_text(json.dumps(names))
(out / "recipe.json").write_text(json.dumps({"seed": seed, "shapes": shapes}))
print("bpp_seed: %d training instances written (seed %d)" % (len(names), seed))
