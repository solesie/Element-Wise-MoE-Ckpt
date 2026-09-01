#!/usr/bin/env python3
"""CPU-only WAF replay with DeepSpeed MoE files plus one persistent log/expert."""

import argparse, glob, io, json, multiprocessing as mp, os, pickle, shutil, tempfile, threading, time
from pathlib import Path

A = 4096


def tenth_aligned(n):
    return max(A, ((int(n) + 5 * A) // (10 * A)) * A)


def load(trace, steps, budget, workload="log"):
    plans, meta, raw = {}, {}, []
    for name in sorted(glob.glob(str(Path(trace) / "waf_rank*.jsonl"))):
        rank, rows = None, {s: [] for s in steps}
        with open(name) as f:
            for line in f:
                x = json.loads(line)
                if x.get("record") == "metadata":
                    rank, meta[int(x["rank"])] = int(x["rank"]), x
                elif x.get("record") == "write_batch" and int(x["step"]) in rows:
                    rows[int(x["step"])].extend(x["writes"])
        if rank is None or any(not rows[s] for s in steps):
            raise RuntimeError(f"missing metadata/step in {name}")
        plans[rank] = rows
    if sorted(plans) != list(range(8)) or {meta[r]["world_size"] for r in meta} != {8}:
        raise RuntimeError("trace must contain ranks 0..7 with world_size=8")

    for r in plans:
        for s in steps:
            raw += [(r, s, *x) for x in plans[r][s]]
    layers = {p: sorted({int(x[4]) for x in raw if x[2] == "y" and meta[x[0]]["pp_rank"] == p})
              for p in {meta[r]["pp_rank"] for r in meta}}

    if workload in ("full_checkpoint_10pct", "full_checkpoint_fixed"):
        if budget is not None:
            raise RuntimeError(f"{workload} does not accept a validation budget")
        active_step = steps[0]
        lfcl_sizes = {int(x[6]) for x in raw if x[2] == "c"}
        if len(lfcl_sizes) != 1:
            raise RuntimeError(f"expected one LFCL size, got {sorted(lfcl_sizes)}")
        lfcl_bytes = lfcl_sizes.pop()
        experts = {}
        for x in raw:
            r, _s, kind, _op, layer, expert, _bytes, _off = x
            if kind != "y":
                continue
            pp = int(meta[r]["pp_rank"])
            key = (r, pp, layers[pp].index(int(layer)), int(expert))
            experts[key] = None
        nonexpert = [x for x in raw if x[1] == active_step and x[2] == "n"]
        if len(experts) != 1536 or len(nonexpert) != 8:
            raise RuntimeError(f"full checkpoint requires 1536 experts + 8 non-expert calls, got {len(experts)} + {len(nonexpert)}")
        out = {r: {active_step: []} for r in plans}
        scale = (lambda _n: A) if workload == "full_checkpoint_fixed" else tenth_aligned
        for r, pp, layer, expert in sorted(experts):
            out[r][active_step].append(("c", "r", pp, layer, expert, scale(lfcl_bytes)))
        for x in nonexpert:
            r, _s, _kind, op, _layer, _expert, n, _off = x
            pp = int(meta[r]["pp_rank"])
            out[r][active_step].append(("n", op, pp, -1, -1, scale(n)))
        source = len(experts) * lfcl_bytes + sum(int(x[6]) for x in nonexpert)
        payload = sum(op[5] for r in out for op in out[r][active_step])
        info = dict(source_bytes=source, payload_bytes=payload,
                    trace_ops=len(experts) + len(nonexpert), physical_ops=len(experts) + len(nonexpert),
                    scale_mode="fixed_4KiB_per_call" if workload == "full_checkpoint_fixed" else "one_tenth",
                    workload=workload,
                    active_steps=[active_step], extrapolation_factor=source / payload,
                    operation_counts={"lfcl": len(experts), "nonexpert": len(nonexpert), "lfsy": 0})
        return out, meta, info

    source = sum(int(x[6]) for x in raw)
    if budget is None:
        sizes = [int(x[6]) for x in raw]
        if any(n <= 0 or n % A for n in sizes):
            raise RuntimeError("full replay requires positive 4 KiB-aligned io_bytes")
    else:
        floor = A * len(raw)
        if budget < floor:
            raise RuntimeError(f"budget must be >= {floor}")
        extra = budget - floor
        sizes = [A + (extra * int(x[6]) // source // A) * A for x in raw]

    out = {r: {s: [] for s in steps} for r in plans}
    for x, n in zip(raw, sizes):
        r, s, kind, op, layer, expert, _bytes, _off = x
        pp = int(meta[r]["pp_rank"])
        if kind == "n":
            # Keep every rank's call.  DeepSpeed names the replicated
            # non-expert files by model-parallel rank, so multiple EP ranks
            # intentionally target the same PP file.
            out[r][s].append((kind, op, pp, -1, -1, n))
        else:
            local_layer = layers[pp].index(int(layer))
            out[r][s].append((kind, op, pp, local_layer, int(expert), n))
    info = dict(source_bytes=source, payload_bytes=sum(sizes), trace_ops=len(raw),
                physical_ops=sum(len(out[r][s]) for r in out for s in steps),
                scale_mode="full" if budget is None else "bounded_validation",
                workload=workload, active_steps=list(steps))
    return out, meta, info


def load_cached(trace, steps, budget, workload, cache):
    key = (str(Path(trace).resolve()), tuple(steps), budget, workload)
    if cache and Path(cache).exists():
        x = pickle.load(open(cache, "rb"))
        if x[0] == key: return x[1]
    value = load(trace, steps, budget, workload)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f: pickle.dump((key, value), f)
        os.replace(tmp, cache)
    return value


def paths(root, op):
    kind, _action, pp, layer, expert, _n = op
    if kind == "n":
        return str(Path(root) / f"mp_rank_{pp:02d}_model_states.pt")
    stem = f"layer_{layer}_expert_{expert}_mp_rank_{pp:02d}"
    suffix = ".lfsy.log" if kind == "y" else "_model_states.pt"
    return str(Path(root) / (stem + suffix))


def seed(root, plans, meta, steps):
    Path(root).mkdir(parents=True, exist_ok=True)
    targets = {paths(root, op) for r in plans for s in steps for op in plans[r][s]}
    logs = {p for p in targets if p.endswith(".lfsy.log")}
    bases = {p.replace(".lfsy.log", "_model_states.pt") for p in logs}
    bases |= {paths(root, op) for r in plans for s in steps for op in plans[r][s] if op[0] == "c"}
    nonexpert = {p for p in targets if Path(p).name.startswith("mp_rank_")}
    optimizer = {str(Path(root) / f"expp_rank_{int(meta[r]['ep_rank'])}_mp_rank_{int(meta[r]['pp_rank']):02d}_optim_states.pt")
                 for r in meta}
    for p in bases | logs | nonexpert | optimizer:
        with open(p, "wb", buffering=0) as f:
            f.write(b"\0" * A)
    return len(bases), len(logs), len(nonexpert), len(optimizer)


def write_all(fd, data):
    while data:
        n = os.write(fd, data)
        if n <= 0:
            raise OSError("short write")
        data = data[n:]


def set_comm(name):
    """Give kernel traces a stable label without adding work inside the write loop."""
    try:
        Path("/proc/self/comm").write_text(name[:15])
    except OSError:
        pass


def worker(groups, root, steps, writer, open_mode, sync, sink, start, done, release):
    handles, files, torch = {}, {}, None
    try:
        ops = [op for r in groups for s in steps for op in groups[r][s]]
        if writer != "noop":
            buf = bytearray(max(op[5] for op in ops))
            views = {n: memoryview(buf)[:n] for *_, n in ops}
            states = {}
            if writer == "torch":
                import torch
                torch.set_num_threads(1)
                states = {n: {"payload": torch.frombuffer(views[n], dtype=torch.uint8)} for n in views}
                # Warm imports and the real ZIP serializer before the timed
                # interval.  ZIP serialization needs a seekable destination.
                torch.save(states[min(states)], io.BytesIO(), _use_new_zipfile_serialization=True)
            if open_mode in ("persistent", "all-persistent"):
                for op in ops:
                    if op[0] == "y" or open_mode == "all-persistent":
                        p = "/dev/null" if sink == "null" else paths(root, op)
                        if p not in handles:
                            append = op[0] == "y"
                            handles[p] = os.open(p, (os.O_WRONLY | os.O_APPEND) if append else os.O_RDWR)
                            if writer == "torch":
                                files[p] = os.fdopen(handles[p], "ab" if append else "r+b", closefd=False)
        done.send_bytes(b"R")
        start.wait()
        rank_names = sorted(groups)
        set_comm(f"waf-rank-{rank_names[0]}" if len(rank_names) == 1 else "waf-proc1")
        for s in steps:
            for r in groups:
                for op in groups[r][s]:
                    if writer == "noop":
                        continue
                    kind, action, _pp, _layer, _expert, n = op
                    p = "/dev/null" if sink == "null" else paths(root, op)
                    close = False
                    if kind == "y":
                        if open_mode in ("persistent", "all-persistent"):
                            fd, f = handles[p], files.get(p)
                            if action == "c":
                                if f: f.flush()
                                os.ftruncate(fd, 0)
                                f.seek(0) if f else os.lseek(fd, 0, os.SEEK_SET)
                        else:
                            mode = "wb" if action == "c" else "ab"
                            f = open(p, mode, buffering=0); fd, close = f.fileno(), True
                        if writer == "torch":
                            torch.save(states[n], f, _use_new_zipfile_serialization=True)
                            if sync in ("flush", "fsync"): f.flush()
                        else:
                            write_all(fd, views[n])
                        if sync == "fsync": os.fsync(fd)
                        if close: f.close()
                    elif open_mode == "all-persistent":
                        fd, f = handles[p], files.get(p)
                        if f:
                            f.flush(); f.seek(0); f.truncate(0)
                            torch.save(states[n], f, _use_new_zipfile_serialization=True)
                            if sync in ("flush", "fsync"): f.flush()
                        else:
                            os.ftruncate(fd, 0); os.lseek(fd, 0, os.SEEK_SET)
                            write_all(fd, views[n])
                        if sync == "fsync": os.fsync(fd)
                    elif writer == "torch":
                        torch.save(states[n], p, _use_new_zipfile_serialization=True)
                        if sync == "fsync":
                            fd = os.open(p, os.O_RDONLY); os.fsync(fd); os.close(fd)
                    else:
                        fd = os.open(p, os.O_WRONLY | os.O_TRUNC)
                        write_all(fd, views[n])
                        if sync == "fsync": os.fsync(fd)
                        os.close(fd)
        set_comm("python3")
        done.send_bytes(b"D")
        release.wait()
    except BaseException as e:
        try: done.send_bytes(("E" + repr(e)).encode())
        except Exception: pass
    finally:
        for f in files.values(): f.close()
        for fd in handles.values():
            try: os.close(fd)
            except OSError: pass


def control(ctl, ack, command):
    if not ctl: return
    with open(ctl, "w", buffering=1) as c: c.write(command + "\n")
    if ack:
        with open(ack) as a:
            if a.readline().strip() != "ack": raise RuntimeError("perf control failed")


def memory_kib(pids):
    total = 0
    for pid in pids:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            total += next(int(x.split()[1]) for x in f if x.startswith("Pss:"))
    return total


def run_process(plans, args, root):
    ctx, start, release = mp.get_context("fork"), None, None
    start, release = ctx.Event(), ctx.Event()
    groups = [plans] if args.executor == "proc1" else [{r: plans[r]} for r in sorted(plans)]
    procs, pipes = [], []
    for group in groups:
        rd, wr = ctx.Pipe(False)
        p = ctx.Process(target=worker, args=(group, root, args.steps, args.writer,
                                            args.open_mode, args.sync, args.sink, start, wr, release))
        p.start(); wr.close(); procs.append(p); pipes.append(rd)
    for rd in pipes:
        if rd.recv_bytes() != b"R": raise RuntimeError("worker setup failed")
    pss = memory_kib([p.pid for p in procs])
    control(args.perf_ctl, args.perf_ack, "enable")
    if args.markers:
        set_comm("waf-control"); os.write(2, b"EXACT_WAF_BEGIN\n"); set_comm("python3")
    t0 = time.perf_counter_ns(); start.set()
    for rd in pipes:
        msg = rd.recv_bytes()
        if msg != b"D": raise RuntimeError(msg.decode(errors="replace"))
    elapsed = (time.perf_counter_ns() - t0) / 1e9
    if args.markers:
        set_comm("waf-control"); os.write(2, b"EXACT_WAF_END\n"); set_comm("python3")
    control(args.perf_ctl, args.perf_ack, "disable")
    release.set()
    for p in procs:
        p.join()
        if p.exitcode: raise RuntimeError(f"worker {p.pid} exited {p.exitcode}")
    return elapsed, pss


def run_threads(plans, args, root):
    start, release, ready, finished = threading.Event(), threading.Event(), [], []
    threads = []
    for r in sorted(plans):
        a, b = os.pipe(); ready.append(a); finished.append(a)
        class Pipe:
            def send_bytes(self, x, w=b): os.write(w, bytes([len(x)]) + x)
        t = threading.Thread(target=worker, args=({r: plans[r]}, root, args.steps, args.writer,
                             args.open_mode, args.sync, args.sink, start, Pipe(), release))
        t.start(); threads.append((t, a, b))
    def recv(fd):
        n = os.read(fd, 1)[0]; return os.read(fd, n)
    for _t, fd, _w in threads:
        if recv(fd) != b"R": raise RuntimeError("thread setup failed")
    pss = memory_kib([os.getpid()])
    if args.markers: os.write(2, b"EXACT_WAF_BEGIN\n")
    t0 = time.perf_counter_ns(); start.set()
    for _t, fd, _w in threads:
        msg = recv(fd)
        if msg != b"D": raise RuntimeError(msg.decode(errors="replace"))
    elapsed = (time.perf_counter_ns() - t0) / 1e9
    if args.markers: os.write(2, b"EXACT_WAF_END\n")
    release.set()
    for t, fd, w in threads: t.join(); os.close(fd); os.close(w)
    return elapsed, pss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True); p.add_argument("--work", required=True)
    p.add_argument("--steps", default="1001,1002,1003")
    p.add_argument("--workload", choices=("log", "full_checkpoint_10pct", "full_checkpoint_fixed"),
                   default="log")
    p.add_argument("--budget", default="full",
                   help="full uses trace io_bytes verbatim; an integer enables bounded validation")
    p.add_argument("--writer", choices=("noop", "raw", "torch"), required=True)
    p.add_argument("--executor", choices=("proc1", "proc8", "thread8"), required=True)
    p.add_argument("--open-mode", choices=("persistent", "all-persistent", "per-write"), default="persistent")
    p.add_argument("--sync", choices=("none", "flush", "fsync"), default="none")
    p.add_argument("--sink", choices=("disk", "null"), default="disk")
    p.add_argument("--plan-cache")
    p.add_argument("--perf-ctl"); p.add_argument("--perf-ack"); p.add_argument("--markers", action="store_true")
    a = p.parse_args(); a.steps = tuple(map(int, a.steps.split(",")))
    budget_text = str(a.budget).strip().lower()
    a.budget = None if budget_text == "full" else int(budget_text)
    if a.budget is not None and a.budget > 1_000_000_000:
        raise SystemExit("bounded validation --budget must be <= 1,000,000,000")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    plans, meta, info = load_cached(a.trace, a.steps, a.budget, a.workload, a.plan_cache)
    a.steps = tuple(info["active_steps"])
    Path(a.work).mkdir(parents=True, exist_ok=True)
    root = tempfile.mkdtemp(prefix="exact-waf-", dir=a.work)
    try:
        topology = seed(root, plans, meta, a.steps)
        elapsed, pss = run_threads(plans, a, root) if a.executor == "thread8" else run_process(plans, a, root)
        stored = sum(x.stat().st_size for x in Path(root).iterdir())
        print(json.dumps({**info, "steps": a.steps, "writer": a.writer, "executor": a.executor,
              "open_mode": a.open_mode, "sync": a.sync, "seconds": elapsed,
              "sink": a.sink,
              "worker_pss_kib": pss,
              "GBps": info["payload_bytes"] / elapsed / 1e9, "stored_bytes": stored,
              "topology": dict(expert_base=topology[0], expert_log=topology[1], nonexpert=topology[2],
                               optimizer=topology[3]),
              "torch": __import__("torch").__version__ if a.writer == "torch" else None}))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__": main()
