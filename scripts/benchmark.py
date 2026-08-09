"""L4 benchmark -- measure latency / throughput / cost after deployment, local vs cloud.

Start the service first, then run:
    python scripts/benchmark.py --url http://localhost:8000   --label local
    python scripts/benchmark.py --url http://<cloud-ip>:8000  --label cloud --hourly-cost 0.526
    python scripts/benchmark.py --compare reports/bench_local.json reports/bench_cloud.json

Produces reports/bench_<label>.json plus a markdown comparison table to paste into the report.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ACNE = {
    "skin_type": "oily", "skin_type_confidence": 0.9,
    "concerns": [{"concern": "acne", "score": 0.88}, {"concern": "large_pores", "score": 0.7},
                 {"concern": "dark_spots", "score": 0.2}, {"concern": "redness", "score": 0.15},
                 {"concern": "wrinkles", "score": 0.05}, {"concern": "dryness", "score": 0.1}],
}
PAYLOAD = {"profile": {"query": "oily skin with acne and large pores", "budget_usd": 40},
           "analysis": ACNE, "top_k": 3}


def _one(url: str, timeout: int) -> tuple[float, bool]:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{url}/recommend", json=PAYLOAD, timeout=timeout)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return (time.perf_counter() - t0) * 1000, ok


def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(url: str, n: int, conc: int, warmup: int, timeout: int) -> dict:
    print(f"target {url}  requests {n}  concurrency {conc}")

    # Cold start: the very first request after the service comes up, recorded on its own
    # because cloud instances routinely show a large cold-start penalty that would otherwise
    # skew the latency percentiles.
    cold_ms, cold_ok = _one(url, timeout)
    print(f"  cold-start request: {cold_ms:.0f} ms  {'OK' if cold_ok else 'FAILED'}")

    for _ in range(warmup):
        _one(url, timeout)
    print(f"  warmup done ({warmup} requests)")

    # --- Sequential: pure latency, unaffected by queueing ---
    seq = [_one(url, timeout) for _ in range(n)]
    seq_ms = [m for m, ok in seq if ok]
    seq_fail = sum(1 for _, ok in seq if not ok)

    # --- Concurrent: throughput ---
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        par = list(ex.map(lambda _: _one(url, timeout), range(n)))
    wall = time.perf_counter() - t0
    par_ms = [m for m, ok in par if ok]
    par_fail = sum(1 for _, ok in par if not ok)
    rps = len(par_ms) / wall if wall > 0 else 0

    return {
        "url": url, "n": n, "concurrency": conc,
        "cold_start_ms": round(cold_ms, 1),
        "latency_ms": {
            "mean": round(statistics.fmean(seq_ms), 1) if seq_ms else None,
            "p50": round(pctl(seq_ms, 0.50), 1), "p95": round(pctl(seq_ms, 0.95), 1),
            "p99": round(pctl(seq_ms, 0.99), 1),
            "min": round(min(seq_ms), 1) if seq_ms else None,
            "max": round(max(seq_ms), 1) if seq_ms else None,
        },
        "throughput_rps": round(rps, 2),
        "concurrent_p95_ms": round(pctl(par_ms, 0.95), 1),
        "failures": {"sequential": seq_fail, "concurrent": par_fail},
        "wall_seconds": round(wall, 2),
    }


def add_cost(res: dict, hourly: float | None) -> dict:
    """Turn the instance hourly price into a cost per 1k requests -- that is the figure the
    report needs, not the raw timings."""
    if hourly and res["throughput_rps"] > 0:
        per_req = hourly / 3600 / res["throughput_rps"]
        res["cost"] = {"instance_usd_per_hour": hourly,
                       "usd_per_1k_requests": round(per_req * 1000, 4)}
    return res


def table(runs: dict[str, dict]) -> str:
    keys = list(runs)
    rows = [
        ("cold start (ms)", lambda r: f"{r['cold_start_ms']:.0f}"),
        ("latency p50 (ms)", lambda r: f"{r['latency_ms']['p50']:.0f}"),
        ("latency p95 (ms)", lambda r: f"{r['latency_ms']['p95']:.0f}"),
        ("latency p99 (ms)", lambda r: f"{r['latency_ms']['p99']:.0f}"),
        ("throughput (req/s)", lambda r: f"{r['throughput_rps']:.2f}"),
        ("p95 under concurrency (ms)", lambda r: f"{r['concurrent_p95_ms']:.0f}"),
        ("failures", lambda r: str(r['failures']['sequential'] + r['failures']['concurrent'])),
        ("cost per 1k requests (USD)", lambda r: (f"{r['cost']['usd_per_1k_requests']:.4f}"
                                                  if r.get("cost") else "—")),
    ]
    out = ["| metric | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, fn in rows:
        out.append(f"| {name} | " + " | ".join(fn(runs[k]) for k in keys) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", default="local", help="local / cloud-t4 / cloud-cpu ...")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--hourly-cost", type=float, default=None,
                    help="hourly price of this instance in USD, used to derive cost per 1k "
                         "requests")
    ap.add_argument("--compare", nargs="+",
                    help="pass several bench_*.json files to render only the comparison table")
    args = ap.parse_args()

    Path("reports").mkdir(exist_ok=True)

    if args.compare:
        runs = {Path(f).stem.replace("bench_", ""): json.loads(Path(f).read_text())
                for f in args.compare}
        md = table(runs)
        print("\n" + md)
        Path("reports/bench_comparison.md").write_text(md + "\n")
        print("\nsaved reports/bench_comparison.md (ready to paste into the report)")
        return

    res = add_cost(run(args.url, args.n, args.concurrency, args.warmup, args.timeout),
                   args.hourly_cost)
    out = Path(f"reports/bench_{args.label}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    print(f"\n  latency p50 {res['latency_ms']['p50']:.0f} ms / "
          f"p95 {res['latency_ms']['p95']:.0f} ms / p99 {res['latency_ms']['p99']:.0f} ms")
    print(f"  throughput {res['throughput_rps']:.2f} req/s (concurrency {args.concurrency})")
    if res.get("cost"):
        print(f"  ${res['cost']['usd_per_1k_requests']:.4f} per 1k requests")
    if sum(res["failures"].values()):
        print(f"  ⚠️ {sum(res['failures'].values())} request(s) failed")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
