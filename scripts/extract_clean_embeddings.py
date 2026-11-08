"""Clean embedding stage extracted from data-audit notebook cells 25-28,40,43-45.
This is the pre-existing shared clean-feature stage consumed by FINAL V2, not pilot results.
Publication adaptations: portable work/ paths; MPS/CUDA/CPU selection; original batch sizes.
It downloads large encoders and recomputes all six arrays. Not run during publication audit.
"""
from pathlib import Path
import pandas as pd
import torch
base_dir=Path(__file__).resolve().parents[1]/"work"
split_dir=base_dir/"splits"
train_df=pd.read_csv(split_dir/"train.csv")
validation_df=pd.read_csv(split_dir/"validation.csv")
test_df=pd.read_csv(split_dir/"test.csv")
device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Original data-audit notebook cell 25
# tes memuat Multilingual E5-large
import numpy as np
from sentence_transformers import SentenceTransformer

model_name = "intfloat/multilingual-e5-large"

text_model = SentenceTransformer(
    model_name,
    device=device
)

sample_texts = [
    f"query: {caption}"
    for caption in train_df["text_caption"].head(4).astype(str)
]

sample_embeddings = text_model.encode(
    sample_texts,
    batch_size=4,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Embedding shape:", sample_embeddings.shape)
print("Data type:", sample_embeddings.dtype)

print("\nEmbedding norms:")
print(np.linalg.norm(sample_embeddings, axis=1))

# Original data-audit notebook cell 26
# buat embedding untuk data train

from pathlib import Path
import numpy as np
import time

embedding_dir = (base_dir / "embeddings" / "clean")
embedding_dir.mkdir(parents=True, exist_ok=True)

train_texts = [
    f"query: {caption}"
    for caption in train_df["text_caption"].astype(str)
]

start_time = time.time()

train_text_embeddings = text_model.encode(
    train_texts,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

elapsed_time = time.time() - start_time

np.save(
    embedding_dir / "train_multilingual_e5_large.npy",
    train_text_embeddings
)

print("Embedding shape:", train_text_embeddings.shape)
print("Data type:", train_text_embeddings.dtype)
print("Elapsed time:", round(elapsed_time / 60, 2), "minutes")
print(
    "File size:",
    round(
        (embedding_dir / "train_multilingual_e5_large.npy").stat().st_size
        / 1024**2,
        2
    ),
    "MB"
)

print(
    "Mean embedding norm:",
    np.linalg.norm(
        train_text_embeddings,
        axis=1
    ).mean()
)

# Original data-audit notebook cell 27
# buat embedding validation
import numpy as np
import time

validation_texts = [
    f"query: {caption}"
    for caption in validation_df["text_caption"].astype(str)
]

start_time = time.time()

validation_text_embeddings = text_model.encode(
    validation_texts,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

elapsed_time = time.time() - start_time

validation_embedding_path = (
    embedding_dir / "validation_multilingual_e5_large.npy"
)

np.save(
    validation_embedding_path,
    validation_text_embeddings
)

print("Embedding shape:", validation_text_embeddings.shape)
print("Data type:", validation_text_embeddings.dtype)
print("Elapsed time:", round(elapsed_time / 60, 2), "minutes")
print(
    "Mean embedding norm:",
    np.linalg.norm(
        validation_text_embeddings,
        axis=1
    ).mean()
)

# Original data-audit notebook cell 28
# buat embedding untuk data test
import numpy as np
import time

test_texts = [
    f"query: {caption}"
    for caption in test_df["text_caption"].astype(str)
]

start_time = time.time()

test_text_embeddings = text_model.encode(
    test_texts,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

elapsed_time = time.time() - start_time

test_embedding_path = (
    embedding_dir / "test_multilingual_e5_large.npy"
)

np.save(
    test_embedding_path,
    test_text_embeddings
)

print("Embedding shape:", test_text_embeddings.shape)
print("Data type:", test_text_embeddings.dtype)
print("Elapsed time:", round(elapsed_time / 60, 2), "minutes")
print(
    "Mean embedding norm:",
    np.linalg.norm(
        test_text_embeddings,
        axis=1
    ).mean()
)

# Original data-audit notebook cell 40
# tes SigLIP 2 pada 2 gambar

import torch
import numpy as np

from PIL import Image
from transformers import AutoProcessor, AutoModel

SIGLIP_MODEL_NAME = "google/siglip2-so400m-patch14-384"

image_device = torch.device(
    device
)

IMAGE_DTYPE = torch.float16 if image_device.type in {"mps", "cuda"} else torch.float32

print("Loading processor...")

siglip_processor = AutoProcessor.from_pretrained(
    SIGLIP_MODEL_NAME
)

print("Loading SigLIP 2...")

siglip_model = AutoModel.from_pretrained(
    SIGLIP_MODEL_NAME,
    dtype=IMAGE_DTYPE,
    low_cpu_mem_usage=True
)

siglip_model = siglip_model.to(image_device)
siglip_model.eval()

# Ambil dua gambar train untuk tes
sample_image_paths = [
    base_dir / image_path
    for image_path in train_df["image_path"].head(2)
]

sample_images = [
    Image.open(image_path).convert("RGB")
    for image_path in sample_image_paths
]

inputs = siglip_processor(
    images=sample_images,
    return_tensors="pt"
)

# Pindahkan input ke MPS.
# Pixel image disamakan dengan dtype model.
inputs = {
    key: (
        value.to(
            image_device,
            dtype=IMAGE_DTYPE
        )
        if key == "pixel_values"
        else value.to(image_device)
    )
    for key, value in inputs.items()
}

with torch.no_grad():
    sample_image_embeddings = (
        siglip_model.get_image_features(**inputs)
    )

# Antisipasi perbedaan output antarversi Transformers
if hasattr(sample_image_embeddings, "pooler_output"):
    sample_image_embeddings = (
        sample_image_embeddings.pooler_output
    )

# Normalisasi agar panjang setiap embedding = 1
sample_image_embeddings = torch.nn.functional.normalize(
    sample_image_embeddings,
    p=2,
    dim=1
)

sample_image_embeddings = (
    sample_image_embeddings
    .float()
    .cpu()
    .numpy()
)

print("\nDevice:", image_device)
print("Embedding shape:", sample_image_embeddings.shape)
print("Data type:", sample_image_embeddings.dtype)
print(
    "Embedding norms:",
    np.linalg.norm(sample_image_embeddings, axis=1)
)

# Original data-audit notebook cell 43
# buat embedding gambar train

# Buat SigLIP 2 image embedding untuk train set

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import time
import gc

embedding_dir = (base_dir / "embeddings" / "clean")
embedding_dir.mkdir(parents=True, exist_ok=True)

train_image_embedding_path = (
    embedding_dir
    / "train_siglip2_so400m_patch14_384.npy"
)

batch_size = 4
train_image_embeddings_list = []

start_time = time.time()

for start_index in tqdm(
    range(0, len(train_df), batch_size),
    desc="Encoding train images"
):
    batch_df = train_df.iloc[
        start_index:start_index + batch_size
    ]

    batch_images = []

    for relative_image_path in batch_df["image_path"]:
        full_image_path = base_dir / relative_image_path

        with Image.open(full_image_path) as image:
            batch_images.append(
                image.convert("RGB").copy()
            )

    inputs = siglip_processor(
        images=batch_images,
        return_tensors="pt"
    )

    inputs = {
        key: (
            value.to(
                image_device,
                dtype=IMAGE_DTYPE
            )
            if key == "pixel_values"
            else value.to(image_device)
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        batch_embeddings = (
            siglip_model.get_image_features(**inputs)
        )

    if hasattr(batch_embeddings, "pooler_output"):
        batch_embeddings = (
            batch_embeddings.pooler_output
        )

    batch_embeddings = torch.nn.functional.normalize(
        batch_embeddings,
        p=2,
        dim=1
    )

    train_image_embeddings_list.append(
        batch_embeddings
        .float()
        .cpu()
        .numpy()
    )

    for image in batch_images:
        image.close()

    del inputs
    del batch_embeddings
    del batch_images

# Gabungkan semua batch
train_image_embeddings = np.concatenate(
    train_image_embeddings_list,
    axis=0
)

assert len(train_image_embeddings) == len(train_df)

np.save(
    train_image_embedding_path,
    train_image_embeddings
)

elapsed_time = time.time() - start_time

print("\n=== TRAIN IMAGE EMBEDDING ===")
print("Embedding shape:", train_image_embeddings.shape)
print("Data type:", train_image_embeddings.dtype)
print("Elapsed time:", round(elapsed_time / 60, 2), "minutes")
print(
    "Mean embedding norm:",
    np.linalg.norm(
        train_image_embeddings,
        axis=1
    ).mean()
)
print(
    "File size:",
    round(
        train_image_embedding_path.stat().st_size
        / 1024**2,
        2
    ),
    "MB"
)
print("Saved to:", train_image_embedding_path)

gc.collect()

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Original data-audit notebook cell 44
# buat embedding gambar validation

# Buat SigLIP 2 image embedding untuk validation set

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import time
import gc

validation_image_embedding_path = (
    embedding_dir
    / "validation_siglip2_so400m_patch14_384.npy"
)

batch_size = 4
validation_image_embeddings_list = []

start_time = time.time()

for start_index in tqdm(
    range(0, len(validation_df), batch_size),
    desc="Encoding validation images"
):
    batch_df = validation_df.iloc[
        start_index:start_index + batch_size
    ]

    batch_images = []

    for relative_image_path in batch_df["image_path"]:
        full_image_path = base_dir / relative_image_path

        with Image.open(full_image_path) as image:
            batch_images.append(
                image.convert("RGB").copy()
            )

    inputs = siglip_processor(
        images=batch_images,
        return_tensors="pt"
    )

    inputs = {
        key: (
            value.to(
                image_device,
                dtype=IMAGE_DTYPE
            )
            if key == "pixel_values"
            else value.to(image_device)
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        batch_embeddings = (
            siglip_model.get_image_features(**inputs)
        )

    if hasattr(batch_embeddings, "pooler_output"):
        batch_embeddings = (
            batch_embeddings.pooler_output
        )

    batch_embeddings = torch.nn.functional.normalize(
        batch_embeddings,
        p=2,
        dim=1
    )

    validation_image_embeddings_list.append(
        batch_embeddings
        .float()
        .cpu()
        .numpy()
    )

    del inputs
    del batch_embeddings
    del batch_images

validation_image_embeddings = np.concatenate(
    validation_image_embeddings_list,
    axis=0
)

assert len(validation_image_embeddings) == len(validation_df)

np.save(
    validation_image_embedding_path,
    validation_image_embeddings
)

elapsed_time = time.time() - start_time

print("\n=== VALIDATION IMAGE EMBEDDING ===")
print("Embedding shape:", validation_image_embeddings.shape)
print("Data type:", validation_image_embeddings.dtype)
print("Elapsed time:", round(elapsed_time / 60, 2), "minutes")
print(
    "Mean embedding norm:",
    np.linalg.norm(
        validation_image_embeddings,
        axis=1
    ).mean()
)
print("Saved to:", validation_image_embedding_path)

gc.collect()

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Original data-audit notebook cell 45
# buat embedding gambar test

# Buat SigLIP 2 image embedding untuk test set

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import time
import gc

test_image_embedding_path = (
    embedding_dir
    / "test_siglip2_so400m_patch14_384.npy"
)

batch_size = 4
test_image_embeddings_list = []

start_time = time.time()

for start_index in tqdm(
    range(0, len(test_df), batch_size),
    desc="Encoding test images"
):
    batch_df = test_df.iloc[
        start_index:start_index + batch_size
    ]

    batch_images = []

    for relative_image_path in batch_df["image_path"]:
        full_image_path = base_dir / relative_image_path

        with Image.open(full_image_path) as image:
            batch_images.append(
                image.convert("RGB").copy()
            )

    inputs = siglip_processor(
        images=batch_images,
        return_tensors="pt"
    )

    inputs = {
        key: (
            value.to(
                image_device,
                dtype=IMAGE_DTYPE
            )
            if key == "pixel_values"
            else value.to(image_device)
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        batch_embeddings = (
            siglip_model.get_image_features(**inputs)
        )

    if hasattr(batch_embeddings, "pooler_output"):
        batch_embeddings = (
            batch_embeddings.pooler_output
        )

    batch_embeddings = torch.nn.functional.normalize(
        batch_embeddings,
        p=2,
        dim=1
    )

    test_image_embeddings_list.append(
        batch_embeddings
        .float()
        .cpu()
        .numpy()
    )

    del inputs
    del batch_embeddings
    del batch_images

test_image_embeddings = np.concatenate(
    test_image_embeddings_list,
    axis=0
)

assert len(test_image_embeddings) == len(test_df)

np.save(
    test_image_embedding_path,
    test_image_embeddings
)

elapsed_time = time.time() - start_time

print("\n=== TEST IMAGE EMBEDDING ===")
print("Embedding shape:", test_image_embeddings.shape)
print("Data type:", test_image_embeddings.dtype)
print("Elapsed time:", round(elapsed_time / 60, 2), "minutes")
print(
    "Mean embedding norm:",
    np.linalg.norm(
        test_image_embeddings,
        axis=1
    ).mean()
)
print("Saved to:", test_image_embedding_path)

gc.collect()

if torch.backends.mps.is_available():
    torch.mps.empty_cache()
