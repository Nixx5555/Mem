"""Download what the Space needs at build time: the code from GitHub, the
ONNX copy of all-MiniLM-L6-v2, and LoCoMo. Plain Python, so the image needs
no git or curl.

    python fetch.py Nixx5555/Mem endtask-full-run
"""
import hashlib
import io
import os
import shutil
import sys
import tarfile
import urllib.request

ONNX_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
# The file the app and its tests were verified with.
ONNX_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
LOCOMO_URL = ("https://raw.githubusercontent.com/snap-research/locomo/main/"
              "data/locomo10.json")
HOME = os.path.expanduser("~")


def get(url):
    print(f"  fetching {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def untar(data, dest, strip):
    """Extract a .tar.gz, dropping its top folder `strip`."""
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if not m.name.startswith(strip + "/"):
                continue
            m.name = m.name[len(strip) + 1:]
            if m.name:
                tf.extract(m, dest, filter="data")


def main(repo, branch):
    code = get(f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}")
    top = tarfile.open(fileobj=io.BytesIO(code)).getmembers()[0].name.split("/")[0]
    dest = os.path.join(HOME, "mem")
    shutil.rmtree(dest, ignore_errors=True)
    untar(code, dest, top)
    app = os.path.join(dest, "memgate")

    onnx = get(ONNX_URL)
    got = hashlib.sha256(onnx).hexdigest()
    if got != ONNX_SHA256:
        sys.exit(f"MiniLM ONNX checksum mismatch: {got}")
    untar(onnx, os.path.join(app, "models", "all-MiniLM-L6-v2-onnx"), "onnx")

    os.makedirs(os.path.join(app, "data", "locomo"), exist_ok=True)
    with open(os.path.join(app, "data", "locomo", "locomo10.json"), "wb") as fh:
        fh.write(get(LOCOMO_URL))
    print("  done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
