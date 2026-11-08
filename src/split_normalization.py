"""Verbatim normalization from data-audit notebook cell 7 (deduplication only)."""
import re,unicodedata
import pandas as pd

def normalize_caption(text):
    """
    Normalisasi untuk mendeteksi caption yang secara substansi sama:
    - Unicode dinormalisasi
    - huruf kecil
    - URL dihapus
    - tanda baca/simbol diubah menjadi spasi
    - spasi berulang dirapikan
    """
    text = "" if pd.isna(text) else str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Sisakan huruf dan angka
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Rapikan whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
