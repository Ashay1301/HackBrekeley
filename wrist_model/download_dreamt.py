"""
download_dreamt.py

Downloads DREAMT v2.2.0 from PhysioNet using parallel wget processes.
Each subject is one CSV file: data_64Hz/SXXX_whole_df.csv

Usage:
    bin/python wrist_model/download_dreamt.py --username ashaypanchal
"""

import argparse
import getpass
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://physionet.org/files/dreamt/2.2.0/data_64Hz"
DIR_URL  = f"{BASE_URL}/?download"


def fetch_directory(username: str, password: str) -> dict[str, int]:
    """
    Fetch the PhysioNet directory listing and return {subject_id: expected_bytes}.
    Falls back to an assumed complete list if the fetch fails.
    """
    cmd = ["wget", "-q", "--timeout=30", "--tries=2",
           f"--user={username}", f"--password={password}",
           "-O", "-", DIR_URL]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        print("[warn] Could not fetch directory listing — using assumed ID list S002–S101")
        return {f"S{str(i).zfill(3)}": 0 for i in range(2, 102)}

    html = result.stdout.decode("utf-8", errors="replace")
    sizes = {}
    for m in re.finditer(r'href="(S\d{3})_whole_df\.csv".*?(\d{6,})', html):
        sid, sz = m.group(1), int(m.group(2))
        sizes[sid] = sz

    if not sizes:
        print("[warn] Directory listing parsed but no subjects found — using assumed list")
        return {f"S{str(i).zfill(3)}": 0 for i in range(2, 102)}

    sorted_ids = sorted(sizes.keys())
    print(f"[info] Directory listing: {len(sizes)} subjects found ({sorted_ids[0]}-{sorted_ids[-1]})")
    return sizes


def is_complete(dest: str, expected_bytes: int) -> bool:
    """File is considered complete if it exists and is ≥95% of the expected size."""
    if not os.path.exists(dest):
        return False
    local = os.path.getsize(dest)
    if expected_bytes > 0:
        return local >= expected_bytes * 0.95
    return local > 10_000


def download_one(sid: str, out_dir: str, username: str, password: str,
                 expected_bytes: int = 0) -> tuple[str, bool, str]:
    url  = f"{BASE_URL}/{sid}_whole_df.csv"
    dest = os.path.join(out_dir, f"{sid}_whole_df.csv")

    if is_complete(dest, expected_bytes):
        return sid, True, "already complete"

    partial = os.path.exists(dest) and os.path.getsize(dest) > 10_000
    action  = "resuming" if partial else "downloading"

    # Use curl without resume — PhysioNet doesn't support HTTP byte ranges
    cmd = [
        "curl",
        "-L",                        # follow redirects
        "--connect-timeout", "30",
        "--max-time", "300",
        "--retry", "2",
        "--retry-delay", "5",
        "--silent", "--show-error",
        "--user", f"{username}:{password}",
        "-o", dest,
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=320)
    except subprocess.TimeoutExpired:
        return sid, False, "timeout after 320s"

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()[-200:]
        if os.path.exists(dest) and os.path.getsize(dest) < 50_000:
            os.remove(dest)  # delete error HTML pages
        return sid, False, stderr or f"curl exit {result.returncode}"

    if not os.path.exists(dest):
        return sid, False, "file missing after curl"

    local = os.path.getsize(dest)

    # Detect 403/error HTML pages saved to disk (always < 50 KB)
    if local < 50_000:
        # Peek at file content to confirm it's HTML, not a real CSV
        with open(dest, "rb") as fh:
            head = fh.read(512).lower()
        if b"<html" in head or b"403" in head or b"forbidden" in head:
            os.remove(dest)
            return sid, False, "HTTP 403 Forbidden — IP rate-limited, wait and retry"

    if not is_complete(dest, expected_bytes):
        return sid, False, f"incomplete: {local // 1024} KB of ~{expected_bytes // 1024} KB"

    return sid, True, f"{local // 1024} KB ({action})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username",   type=str, default="ashaypanchal")
    parser.add_argument("--out_dir",    type=str, default="data/dreamt/raw")
    parser.add_argument("--workers",    type=int, default=2,
                        help="Parallel downloads — PhysioNet rate-limits above ~3 (default 2)")
    parser.add_argument("--n_subjects", type=int, default=100,
                        help="Max subjects to download (default 100)")
    args = parser.parse_args()

    password = getpass.getpass(f"PhysioNet password for {args.username}: ")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Get real subject list + expected sizes from directory listing ────────────
    print("Fetching directory listing from PhysioNet...")
    all_subjects = fetch_directory(args.username, password)
    subject_ids  = sorted(all_subjects.keys())[: args.n_subjects]
    print(f"Target: {len(subject_ids)} subjects ({subject_ids[0]}–{subject_ids[-1]})")

    # ── Check which need downloading / resuming ──────────────────────────────────
    pending = [
        sid for sid in subject_ids
        if not is_complete(
            os.path.join(args.out_dir, f"{sid}_whole_df.csv"),
            all_subjects[sid],
        )
    ]

    complete_already = len(subject_ids) - len(pending)
    if complete_already:
        print(f"Already complete: {complete_already} subjects (skipped)")

    # Show partial files (will be resumed)
    partial = [
        sid for sid in pending
        if os.path.exists(os.path.join(args.out_dir, f"{sid}_whole_df.csv"))
        and os.path.getsize(os.path.join(args.out_dir, f"{sid}_whole_df.csv")) > 10_000
    ]
    if partial:
        print(f"Will resume {len(partial)} partial downloads")

    if not pending:
        print("All subjects complete!")
        return

    total_gb = sum(all_subjects[s] for s in pending if all_subjects[s] > 0) / 1e9
    print(f"Downloading/resuming {len(pending)} subjects (~{total_gb:.1f} GB) with {args.workers} workers...")

    done, failed = 0, []
    total = len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one, sid, args.out_dir, args.username, password, all_subjects[sid]
            ): sid
            for sid in pending
        }
        for future in as_completed(futures):
            sid, ok, msg = future.result()
            if ok:
                done += 1
                print(f"  [{done}/{total}] {sid} ✓ {msg}")
            else:
                failed.append(sid)
                print(f"  [FAIL] {sid} — {msg}")

    print(f"\nDone. {done} complete, {len(failed)} failed.")
    if failed:
        print(f"Failed: {failed}")
        print("Re-run to retry (partial files will be resumed with -c).")
        sys.exit(1)

    print(f"\nNext step:")
    print(f"  bin/python wrist_model/prepare_dreamt.py --raw_dir {args.out_dir} --out_dir data/dreamt/processed")


if __name__ == "__main__":
    main()
