"""Verbatim FINAL V2 perturbation functions with locked published configuration."""
from pathlib import Path
import json,re
import pandas as pd
from PIL import Image,ImageFilter
_CONFIG=Path(__file__).resolve().parents[1]/"results/final_v2/configs"
_map=json.loads((_CONFIG/"text_obfuscation_matcher_map.json").read_text())
final_lexicon=_map["exact_terms"]
approved_compound_map={k:set(v) for k,v in _map["approved_compound_map"].items()}


def resolve_overlapping_matches(
    candidates
):
    """
    Memilih match tanpa overlap.

    Prioritas:
    1. term yang lebih panjang;
    2. posisi lebih awal;
    3. nama term.
    """

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            -(
                item["end"]
                - item["start"]
            ),
            item["start"],
            item["term"],
        )
    )

    selected_matches = []
    occupied_positions = set()

    for candidate in sorted_candidates:
        candidate_positions = set(
            range(
                candidate["start"],
                candidate["end"]
            )
        )

        if (
            occupied_positions
            & candidate_positions
        ):
            continue

        selected_matches.append(
            candidate
        )

        occupied_positions.update(
            candidate_positions
        )

    return sorted(
        selected_matches,
        key=lambda item: item["start"]
    )


def find_final_text_matches(text):
    """
    Menemukan exact term dan approved compound term.
    """

    text = (
        ""
        if pd.isna(text)
        else str(text)
    )

    candidates = []

    # Exact matching:
    # slot88 cocok, Deportes tidak dianggap depo.
    for term in final_lexicon:
        exact_pattern = re.compile(
            rf"(?<![A-Za-z])"
            rf"{re.escape(term)}"
            rf"(?![A-Za-z])",
            flags=re.IGNORECASE
        )

        for match in exact_pattern.finditer(text):
            candidates.append({
                "term": term,
                "match_type": "exact",
                "matched_text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })

    # Compound matching dilakukan per token utuh.
    token_matches = list(
        re.finditer(
            r"[A-Za-z0-9_]+",
            text
        )
    )

    for token_match in token_matches:
        token_text = token_match.group(0)
        token_lower = token_text.lower()

        for term, approved_tokens in (
            approved_compound_map.items()
        ):
            if token_lower not in approved_tokens:
                continue

            term_pattern = re.compile(
                re.escape(term),
                flags=re.IGNORECASE
            )

            for local_match in (
                term_pattern.finditer(
                    token_text
                )
            ):
                candidates.append({
                    "term": term,
                    "match_type": (
                        "approved_compound"
                    ),
                    "matched_text": (
                        local_match.group(0)
                    ),
                    "start": (
                        token_match.start()
                        + local_match.start()
                    ),
                    "end": (
                        token_match.start()
                        + local_match.end()
                    ),
                    "compound_token": (
                        token_text
                    ),
                })

    return resolve_overlapping_matches(
        candidates
    )


def apply_full_character_period_obfuscation(
    text
):
    """
    Contoh:
    slot   -> s.l.o.t
    gacor  -> g.a.c.o.r
    """

    text = (
        ""
        if pd.isna(text)
        else str(text)
    )

    matches = find_final_text_matches(
        text
    )

    obfuscated_text = text

    # Ganti dari kanan ke kiri agar indeks tidak bergeser.
    for match in reversed(matches):
        original_segment = (
            obfuscated_text[
                match["start"]:
                match["end"]
            ]
        )

        transformed_segment = ".".join(
            original_segment
        )

        obfuscated_text = (
            obfuscated_text[
                :match["start"]
            ]
            + transformed_segment
            + obfuscated_text[
                match["end"]:
            ]
        )

    return {
        "original_text": text,
        "obfuscated_text": obfuscated_text,
        "text_changed": (
            obfuscated_text != text
        ),
        "matched_terms": [
            match["term"]
            for match in matches
        ],
        "unique_matched_terms": sorted({
            match["term"]
            for match in matches
        }),
        "match_types": [
            match["match_type"]
            for match in matches
        ],
        "target_occurrence_count": len(
            matches
        ),
        "matches": matches,
    }


def apply_gaussian_blur(
    image,
    radius
):
    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


def apply_pixelation(
    image,
    scale
):
    if not 0 < scale < 1:
        raise ValueError(
            "Pixelation scale harus "
            "berada antara 0 dan 1."
        )

    original_width, original_height = (
        image.size
    )

    reduced_width = max(
        1,
        round(
            original_width * scale
        )
    )

    reduced_height = max(
        1,
        round(
            original_height * scale
        )
    )

    reduced_image = image.resize(
        (
            reduced_width,
            reduced_height,
        ),
        resample=Image.Resampling.BOX
    )

    pixelated_image = reduced_image.resize(
        (
            original_width,
            original_height,
        ),
        resample=Image.Resampling.NEAREST
    )

    return pixelated_image


def apply_blur_then_pixelation(
    image,
    blur_radius=5,
    pixelation_scale=0.25
):
    blurred_image = apply_gaussian_blur(
        image=image,
        radius=blur_radius
    )

    return apply_pixelation(
        image=blurred_image,
        scale=pixelation_scale
    )
