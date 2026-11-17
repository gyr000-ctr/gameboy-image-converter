"""Tests for the tile-mirror mapping API added to gb_pipeline.py.

Covers the map-editor tile-reuse contract: a tile stored once and referenced
in up to four orientations (index + independent h/v flip bits), the mapping
serialized size for four sample assets, pixel-exact reconstruction from the
mapping alone, the size difference with mirroring disabled, and the error
surfaced when a serialized mapping is corrupted.

The four samples are built here (small, deterministic, no fixtures on disk):
two are mirror-heavy, one has no reusable tiles, and one mixes the two. They
are also used by the size comparison reported with the change.
"""

import hashlib
import struct

import numpy as np
import pytest
from PIL import Image

import gb_pipeline as gp


# ---------------------------------------------------------------------------
# Sample assets (small, deterministic)


def _orient(pat: np.ndarray, o: str) -> np.ndarray:
    if "h" in o:
        pat = np.fliplr(pat)
    if "v" in o:
        pat = np.flipud(pat)
    return pat


# RGB555-stable shades, so samples can be fed straight to convert_for_hardware.
RAMP = np.array(
    [[255, 255, 255], [197, 197, 197], [132, 132, 132], [66, 66, 66]],
    dtype=np.uint8,
)


def _stamp(img, tr, tc, pat):
    img[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8] = RAMP[pat]


def _unique_motifs(count: int, seed: int) -> list:
    """``count`` patterns pairwise distinct under any of the four orientations."""
    rng = np.random.RandomState(seed)
    seen = set()
    motifs = []
    while len(motifs) < count:
        pat = rng.randint(0, 4, size=(8, 8)).astype(np.uint8)
        keys = {gp.pattern_to_2bpp(_orient(pat, o)) for o in ("", "h", "v", "hv")}
        if keys & seen:
            continue
        seen |= keys
        motifs.append(pat)
    return motifs


def sample_hflip_heavy() -> np.ndarray:
    """10x6: 12 motifs, each stored once and reused mostly with hflip."""
    th, tw = 6, 10
    img = np.zeros((th * 8, tw * 8, 3), dtype=np.uint8)
    motifs = _unique_motifs(12, seed=1)
    layout = []
    for i in range(12):
        # one identity copy, four horizontally mirrored references
        layout.extend([(i, o) for o in ("", "h", "h", "h", "h")])
    for k, (i, o) in enumerate(layout[: th * tw]):
        tr, tc = divmod(k, tw)
        _stamp(img, tr, tc, _orient(motifs[i], o))
    return img


def sample_vflip_heavy() -> np.ndarray:
    """8x8: 16 motifs, each stored once and reused mostly with vflip."""
    th, tw = 8, 8
    img = np.zeros((th * 8, tw * 8, 3), dtype=np.uint8)
    motifs = _unique_motifs(16, seed=2)
    layout = []
    for i in range(16):
        layout.extend([(i, o) for o in ("", "v", "v", "v")])
    for k, (i, o) in enumerate(layout[: th * tw]):
        tr, tc = divmod(k, tw)
        _stamp(img, tr, tc, _orient(motifs[i], o))
    return img


def sample_no_reuse() -> np.ndarray:
    th, tw = 6, 8
    img = np.zeros((th * 8, tw * 8, 3), dtype=np.uint8)
    motifs = _unique_motifs(th * tw, seed=3)
    for k, pat in enumerate(motifs):
        tr, tc = divmod(k, tw)
        _stamp(img, tr, tc, pat)
    return img


def sample_mixed() -> np.ndarray:
    """6x12: half the cells reuse 9 motifs in all four orientations, the rest
    are unique tiles that mirroring cannot help."""
    th, tw = 6, 12
    img = np.zeros((th * 8, tw * 8, 3), dtype=np.uint8)
    shared = _unique_motifs(9, seed=4)
    unique = _unique_motifs(36, seed=5)
    for k in range(th * tw):
        tr, tc = divmod(k, tw)
        if k < 36:
            _stamp(img, tr, tc, _orient(shared[k // 4], ["", "h", "v", "hv"][k % 4]))
        else:
            _stamp(img, tr, tc, unique[k - 36])
    return img


SAMPLES = {
    "hflip-heavy": sample_hflip_heavy,
    "vflip-heavy": sample_vflip_heavy,
    "no-reuse": sample_no_reuse,
    "mixed": sample_mixed,
}


def _gb_from_mapping_patterns(patterns, tilemap, hflip=None, vflip=None, pal=None):
    th, tw = tilemap.shape
    if pal is None:
        pal = RAMP[None]
    return gp.GBImage(
        patterns=np.asarray(patterns, dtype=np.uint8),
        tilemap=np.asarray(tilemap, dtype=np.int32),
        attrs_palette=np.zeros((th, tw), dtype=np.uint8),
        attrs_hflip=(np.zeros((th, tw), dtype=bool) if hflip is None else hflip),
        attrs_vflip=(np.zeros((th, tw), dtype=bool) if vflip is None else vflip),
        palettes=pal,
    )


def _gb_from_sample(name: str) -> gp.GBImage:
    """A GBImage holding each sample cell as its own fresh, unflipped tile.

    This isolates the mapping stage from the conversion pipeline's budget
    merge / palette conform, so the size comparison measures mirroring alone.
    """
    img = SAMPLES[name]()
    th, tw = img.shape[0] // 8, img.shape[1] // 8
    ramp_key = {tuple(c): i for i, c in enumerate(RAMP)}
    patterns = np.zeros((th * tw, 8, 8), dtype=np.uint8)
    for tr in range(th):
        for tc in range(tw):
            block = img[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8]
            for y in range(8):
                for x in range(8):
                    patterns[tr * tw + tc, y, x] = ramp_key[tuple(block[y, x])]
    return gp.GBImage(
        patterns=patterns,
        tilemap=np.arange(th * tw, dtype=np.int32).reshape(th, tw),
        attrs_palette=np.zeros((th, tw), dtype=np.uint8),
        attrs_hflip=np.zeros((th, tw), dtype=bool),
        attrs_vflip=np.zeros((th, tw), dtype=bool),
        palettes=RAMP[None],
    )


# ---------------------------------------------------------------------------
# 1. Mapping size for the four sample assets


@pytest.mark.parametrize("name", list(SAMPLES))
def test_mapping_sizes_for_each_sample(name):
    """Every sample yields a mapping whose bytes round-trip and whose map
    entries fit the stored tileset."""
    gb = _gb_from_sample(name)
    mapping = gp.build_tile_mapping(gb, allow_flips=True)
    assert mapping.n_tiles <= mapping.n_cells
    assert mapping.tiles.shape == (mapping.n_tiles, 8, 8)

    payload = mapping.to_bytes()
    written = gp.mapping_size_bytes(mapping)
    assert written == len(payload)
    # 18-byte header + 16 bytes/tile + 12 bytes/palette + 3 bytes/cell
    # (tile index + flip flags) + 1 byte/cell (palette id) + 4-byte CRC
    assert written == (
        18 + 16 * mapping.n_tiles + 12 * mapping.palettes.shape[0]
        + 3 * mapping.n_cells + mapping.n_cells + 4
    )

    decoded = gp.TileMapping.from_bytes(payload)
    assert np.array_equal(decoded.tiles, mapping.tiles)
    assert np.array_equal(decoded.tilemap, mapping.tilemap)
    assert np.array_equal(decoded.hflip, mapping.hflip)
    assert np.array_equal(decoded.vflip, mapping.vflip)


def test_mirroring_shrinks_mapping_vs_unmirrored():
    """Mirror-heavy samples shrink; the no-reuse sample cannot shrink."""
    sizes = {}
    for name in SAMPLES:
        mapping = gp.build_tile_mapping(_gb_from_sample(name), allow_flips=True)
        sizes[name] = (gp.mapping_size_bytes(mapping),
                       gp.unmirrored_size_bytes(mapping))
        assert sizes[name][1] >= sizes[name][0]
    assert sizes["hflip-heavy"][0] < sizes["hflip-heavy"][1]
    assert sizes["vflip-heavy"][0] < sizes["vflip-heavy"][1]
    assert sizes["mixed"][0] < sizes["mixed"][1]
    assert sizes["no-reuse"][0] == sizes["no-reuse"][1]


# ---------------------------------------------------------------------------
# 2. Pixel-exact reconstruction from the mapping


@pytest.mark.parametrize("name", list(SAMPLES))
def test_rebuild_from_mapping_is_pixel_exact(name):
    gb = _gb_from_sample(name)
    mapping = gp.build_tile_mapping(gb, allow_flips=True)
    rebuilt = gp.rebuild_from_mapping(mapping)
    assert np.array_equal(np.asarray(rebuilt), np.asarray(gp.render(gb)))


@pytest.mark.parametrize("name", list(SAMPLES))
def test_rebuild_through_serialized_bytes_is_pixel_exact(name):
    """The map-editor path is bytes-only: serialize, parse, rebuild."""
    gb = _gb_from_sample(name)
    mapping = gp.build_tile_mapping(gb, allow_flips=True)
    decoded = gp.TileMapping.from_bytes(mapping.to_bytes())
    rebuilt = gp.rebuild_from_mapping(decoded)
    assert np.array_equal(np.asarray(rebuilt), np.asarray(gp.render(gb)))


def test_mirrored_cells_in_hflip_sample_are_actually_mirrored():
    mapping = gp.build_tile_mapping(_gb_from_sample("hflip-heavy"))
    assert mapping.n_mirrored_cells > 0


def test_mapping_is_idempotent_when_rebuilt():
    """Mapping each cell's effective content again must not grow the tileset."""
    first = gp.build_tile_mapping(_gb_from_sample("mixed"), allow_flips=True)
    index = gp.apply_tile_mapping(first)
    th, tw = first.tilemap.shape
    patterns = index.reshape(th, 8, tw, 8).transpose(0, 2, 1, 3).reshape(
        th * tw, 8, 8
    )
    gb = _gb_from_mapping_patterns(
        patterns,
        np.arange(th * tw, dtype=np.int32).reshape(th, tw),
        pal=first.palettes,
    )
    second = gp.build_tile_mapping(gb, allow_flips=True)
    assert second.n_tiles == first.n_tiles
    assert np.array_equal(
        np.asarray(gp.rebuild_from_mapping(second)),
        np.asarray(gp.rebuild_from_mapping(first)),
    )


# ---------------------------------------------------------------------------
# 3. Corrupt mapping bytes raise an actionable error


def _sample_mapping():
    return gp.build_tile_mapping(_gb_from_sample("mixed"), allow_flips=True)


def test_corrupt_crc_raises():
    m = _sample_mapping()
    bad = bytearray(m.to_bytes())
    bad[-1] ^= 0xFF
    with pytest.raises(gp.TileMappingError, match="checksum"):
        gp.TileMapping.from_bytes(bytes(bad))


def test_corrupt_tile_byte_with_stale_crc_raises():
    """Flipping a stored-tile byte but keeping the uploaded CRC must fail."""
    m = _sample_mapping()
    raw = bytearray(m.to_bytes())
    raw[18] ^= 0x01  # first byte of the first stored tile
    with pytest.raises(gp.TileMappingError, match="checksum"):
        gp.TileMapping.from_bytes(bytes(raw))


def test_truncated_payload_raises():
    m = _sample_mapping()
    raw = m.to_bytes()
    with pytest.raises(gp.TileMappingError, match="bytes; header"):
        gp.TileMapping.from_bytes(raw[:-7])


def test_bad_magic_raises():
    m = _sample_mapping()
    raw = bytearray(m.to_bytes())
    raw[0:4] = b"XXXX"
    with pytest.raises(gp.TileMappingError, match="magic"):
        gp.TileMapping.from_bytes(bytes(raw))


def test_dangling_tile_reference_raises():
    """A cell that points past the tileset is rejected even with a valid CRC."""
    m = gp.TileMapping(
        tiles=np.zeros((2, 8, 8), dtype=np.uint8),
        tilemap=np.zeros((1, 2), dtype=np.int32),
        hflip=np.zeros((1, 2), dtype=bool),
        vflip=np.zeros((1, 2), dtype=bool),
        attrs_palette=np.zeros((1, 2), dtype=np.uint8),
        palettes=RAMP[None],
    )
    raw = bytearray(m.to_bytes())
    # Each cell reference is (uint16 tile index, uint8 flags); rewrite the
    # *first* cell's index to point past the 2 stored tiles, then re-seal.
    ref_base = 18 + 16 * 2 + 12 * 1
    raw[ref_base : ref_base + 2] = struct.pack(">H", 9)
    import zlib

    body = bytes(raw[:-4])
    raw[-4:] = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    with pytest.raises(gp.TileMappingError, match="references tile"):
        gp.TileMapping.from_bytes(bytes(raw))


def test_zero_tiles_is_rejected_on_construction():
    with pytest.raises(gp.TileMappingError):
        gp.TileMapping(
            tiles=np.zeros((0, 8, 8), dtype=np.uint8),
            tilemap=np.zeros((1, 1), dtype=np.int32),
            hflip=np.zeros((1, 1), dtype=bool),
            vflip=np.zeros((1, 1), dtype=bool),
            attrs_palette=np.zeros((1, 1), dtype=np.uint8),
            palettes=RAMP[None],
        )


# ---------------------------------------------------------------------------
# 4. Optics: conversion output is unchanged by mapping, flips preserve pixels


@pytest.mark.parametrize("name", list(SAMPLES))
@pytest.mark.parametrize("preset", ["color_only", "mono", "logo_color", "logo_mono"])
def test_mapping_does_not_change_conversion_output(name, preset):
    img = Image.fromarray(SAMPLES[name](), "RGB")
    with_mapping = gp.convert_for_hardware(img, preset, carry_tile_mapping=True)
    without = gp.convert_for_hardware(img, preset, carry_tile_mapping=False)
    assert hashlib.sha256(with_mapping.image.tobytes()).digest() == hashlib.sha256(
        without.image.tobytes()
    ).digest()
    assert without.tile_mapping is None


def test_render_of_flipped_cell_equals_flipped_render():
    """Mirror bits are a reference rewrite: placing tile T with hflip must
    equal storing already-flipped pixels, never a change to the tile set."""
    tile = _unique_motifs(1, seed=7)[0]
    gb = _gb_from_mapping_patterns(
        np.stack([tile]),
        np.zeros((1, 2), dtype=np.int32),
        hflip=np.array([[False, True]], dtype=bool),
    )
    out = np.asarray(gp.apply_tile_mapping(gp.build_tile_mapping(gb)))
    assert np.array_equal(out[:, 0:8], tile)
    assert np.array_equal(out[:, 8:16], np.fliplr(tile))


def test_disabling_flips_forces_one_copy_per_orientation():
    tile = _unique_motifs(1, seed=9)[0]
    gb = _gb_from_mapping_patterns(
        np.stack([tile]),
        np.zeros((1, 2), dtype=np.int32),
        hflip=np.array([[False, True]], dtype=bool),
    )
    with_flips = gp.build_tile_mapping(gb, allow_flips=True)
    without = gp.build_tile_mapping(gb, allow_flips=False)
    assert with_flips.n_tiles == 1
    assert with_flips.n_mirrored_cells == 1
    assert without.n_tiles == 2
    assert without.n_mirrored_cells == 0
    # Either way the placed pixels are identical.
    assert np.array_equal(
        gp.apply_tile_mapping(with_flips), gp.apply_tile_mapping(without)
    )


@pytest.mark.parametrize("name", list(SAMPLES))
@pytest.mark.parametrize("preset", ["color_only", "mono", "logo_color", "logo_mono"])
def test_mapping_reproduces_shipped_png(name, preset):
    """The mapping carried by a conversion must rebuild its *shipped* PNG,
    including when the GB Studio palette conform recolored tiles."""
    img = Image.fromarray(SAMPLES[name](), "RGB")
    res = gp.convert_for_hardware(img, preset, carry_tile_mapping=True)
    rebuilt = np.asarray(gp.rebuild_from_mapping(res.tile_mapping))
    assert np.array_equal(rebuilt, np.asarray(res.image))


def test_map_editor_roundtrip_through_file_bytes(tmp_path):
    """End-to-end map-editor path: convert, write mapping bytes to a file,
    read them back, rebuild, and compare to the shipped PNG."""
    img = Image.fromarray(SAMPLES["vflip-heavy"](), "RGB")
    res = gp.convert_for_hardware(img, "mono", carry_tile_mapping=True)
    path = tmp_path / "tiles.map"
    path.write_bytes(res.tile_mapping.to_bytes())
    loaded = gp.TileMapping.from_bytes(path.read_bytes())
    assert np.array_equal(
        np.asarray(gp.rebuild_from_mapping(loaded)), np.asarray(res.image)
    )


def test_mapping_from_conversion_raises_when_disabled():
    img = Image.fromarray(SAMPLES["no-reuse"](), "RGB")
    res = gp.convert_for_hardware(img, "mono", carry_tile_mapping=False)
    with pytest.raises(gp.TileMappingError, match="did not carry"):
        gp.mapping_from_conversion(res)
