"""Merge the repeats of one or more result folders into one reference folder.

Use it when a GPU had to be re-run (for example because Modal placed some "H100" repeats on an
H200): every repeat whose device does not match experiment_config.EXPECTED_DEVICE is dropped, the
medians and ranges are recomputed from the repeats that remain, and the traces are taken from a
source whose traces were captured on the right device.

    python case_study/merge_results.py OUT_DIR SRC_DIR [SRC_DIR ...] [--max N]

--max N keeps the first N usable repeats per GPU (default 5, the book's convention).

Later sources win for traces and all-reduce files; repeats from all sources are pooled.
"""
import gzip
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate  # noqa: E402
import experiment_config as cfg  # noqa: E402
import make_tables  # noqa: E402
import principles  # noqa: E402


def _tag(g):
    return g.lower().replace("-", "_").replace("!", "")


def file_device(f, tdir, fallback=None):
    """Device a trace file was captured on: Chrome traces carry deviceProperties; Nsight and text
    files inherit the experiments container's device (DEVICE.txt) when present."""
    if f.name.endswith(".json.gz"):
        try:
            props = json.load(gzip.open(f, "rt")).get("deviceProperties") or []
            if props:
                return props[0].get("name")
        except Exception:  # noqa: BLE001
            pass
    if "chapter02_lab" in f.parts:
        return None
    p = tdir / "DEVICE.txt"
    if p.exists():
        return p.read_text().strip()
    sib = [x for x in f.parent.glob("*.json.gz")]
    if sib and not f.name.endswith(".json.gz"):
        return file_device(sib[0], tdir, fallback)
    return fallback   # traces are captured by repeat 1's experiments container


def trace_device(tdir):
    p = tdir / "DEVICE.txt"
    if p.exists():
        return p.read_text().strip()
    for f in tdir.rglob("*.json.gz"):
        try:
            d = json.load(gzip.open(f, "rt"))
            props = d.get("deviceProperties") or []
            if props:
                return props[0].get("name", "?")
        except Exception:  # noqa: BLE001
            continue
    return "?"


def main(out, srcs, max_n=5):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "repeats").mkdir(exist_ok=True)
    by_gpu, runs_by_gpu, log = {}, {}, []
    for g in cfg.GPUS:
        tag, want = _tag(g), cfg.EXPECTED_DEVICE.get(g, "")
        runs, labdev = [], []
        for si, src in enumerate(srcs):
            for f in sorted((src / "repeats").glob(f"{tag}_r*.json")):
                r = json.loads(f.read_text())
                r["source"] = f"{src.name}/{f.name}"
                runs.append(r)
                k = len(runs)
                (out / "repeats" / f"{tag}_r{k}.json").write_text(json.dumps(r, indent=2))
                nb = src / "repeats" / f.name.replace(f"{tag}_r", f"chapter02_lab_executed_{tag}_r").replace(".json", ".ipynb")
                labdev.append(aggregate.lab_device(r, nb.read_text() if nb.exists() else None))
                if nb.exists():
                    shutil.copy(nb, out / "repeats" / f"chapter02_lab_executed_{tag}_r{k}.ipynb")
                    if k == 1:
                        shutil.copy(nb, out / f"chapter02_lab_executed_{tag}.ipynb")
        if not runs:
            log.append(f"{g}: no usable repeats"); continue
        merged, sp, views, mlog = aggregate.merge_repeats(runs, want, labdev, max_n=max_n)
        log += [f"{g}: {m}" for m in mlog]
        log.append(f"{g}: {merged['repeats']} experiment repeats, {merged['lab_repeats']} lab repeats kept")
        (out / f"{tag}.json").write_text(json.dumps(merged, indent=2))
        if sp:
            (out / f"{tag}_spread.json").write_text(json.dumps(sp, indent=2))
        by_gpu[g], runs_by_gpu[g] = merged, views
        dest = out / "traces" / tag
        shutil.rmtree(dest, ignore_errors=True)
        for src in srcs:                       # later sources overwrite earlier ones
            tdir = src / "traces" / tag
            if not tdir.exists():
                continue
            for f in tdir.rglob("*"):
                if not f.is_file() or f.name == "DEVICE.txt":
                    continue
                r1 = src / "repeats" / f"{tag}_r1.json"
                fb = json.loads(r1.read_text())["env"]["gpu"] if r1.exists() else None
                dev = file_device(f, tdir, fb)
                if want and dev and want not in dev:
                    log.append(f"{g}: skipped trace {src.name}/{f.relative_to(tdir)} (captured on {dev})"); continue
                (dest / f.relative_to(tdir)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest / f.relative_to(tdir))
    for src in srcs:
        for f in src.glob("allreduce_*.json"):
            r = json.loads(f.read_text())
            dev = (r.get("env") or {}).get("gpu", "")
            g = f.stem.removeprefix("allreduce_").split("x")[0]
            want = next((v for k, v in cfg.EXPECTED_DEVICE.items() if _tag(k) == g), "")
            if want and want not in dev:
                log.append(f"all-reduce: skipped {src.name}/{f.name} (ran on {dev})"); continue
            shutil.copy(f, out / f.name)
            t = src / "traces" / f.stem.removeprefix("allreduce_")
            if t.exists():
                shutil.rmtree(out / "traces" / t.name, ignore_errors=True)
                shutil.copytree(t, out / "traces" / t.name)
    rows = principles.evaluate(by_gpu)
    report = principles.markdown_report(rows, by_gpu)
    counts = principles.count_across_repeats(runs_by_gpu)
    report += principles.markdown_counts(counts)
    for r in rows:
        r["holds_in_repeats"] = counts.get(r["id"], {})
    (out / "principles_report.md").write_text(report)
    (out / "principles_report.json").write_text(json.dumps(rows, indent=2))
    T = make_tables.build(out)
    (out / "master_tables.md").write_text(make_tables.to_md(T))
    (out / "master_tables.json").write_text(json.dumps(T, indent=2))
    (out / "MERGE_LOG.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 5
    if "--max" in args:
        i = args.index("--max"); n = int(args[i + 1]); del args[i:i + 2]
    main(args[0], [pathlib.Path(p) for p in args[1:]], max_n=n)
