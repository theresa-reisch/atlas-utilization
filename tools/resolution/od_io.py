"""Open ATLAS Open Data files from compute nodes.

Compute nodes reach the internet only through the HTTP proxy in $https_proxy,
which xrootd cannot use, and uproot's HTTPS range reads through it fail
intermittently. Files are therefore downloaded whole (curl honours the proxy,
~19 MB/s) to a node-local or scratch directory, read, and deleted.
"""
import contextlib
import os
import subprocess
import tempfile

import uproot


def https_url(url):
    if url.startswith("root://eospublic.cern.ch"):
        return "https://opendata.cern.ch/eos/" + url.split("/eos/", 1)[1]
    return url


@contextlib.contextmanager
def local_tree(url, tree="CollectionTree", retries=3):
    base = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR_DL") or os.path.expanduser("~/scratch/resonex_resolution/dl")
    os.makedirs(base, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".root", dir=base)
    os.close(fd)
    try:
        for attempt in range(retries):
            rc = subprocess.call(["curl", "-sS", "-f", "--retry", "3", "-o", path, https_url(url)])
            if rc == 0:
                break
            print(f"download failed (rc={rc}, attempt {attempt + 1}) for {url}", flush=True)
        else:
            raise OSError(f"could not download {url}")
        with uproot.open(path) as f:
            yield f[tree]
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
