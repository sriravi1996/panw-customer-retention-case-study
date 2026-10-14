"""Download and verify the saved synthetic data release, then extract safely."""
from pathlib import Path
import hashlib, tarfile, tempfile, urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/sriravi1996/panw-customer-retention-case-study/releases/download/v1.0.0/synthetic-data-v1.tar.gz"
SHA256 = "fe57f07e5250a1c61cc811dbb7b97ad9ca6d6267157195c6733aa5e69a1e083c"

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="panw-synthetic-") as temp:
        archive = Path(temp) / "synthetic-data-v1.tar.gz"
        print("Downloading synthetic data...")
        urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Download checksum mismatch")
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                target = (ROOT / member.name).resolve()
                if not target.is_relative_to(ROOT) or not member.isfile():
                    raise ValueError("Unexpected archive entry")
            source.extractall(ROOT, filter="data")
    print("Verified data saved under data/raw, data/cleaned, data/analysis, and eval.")
