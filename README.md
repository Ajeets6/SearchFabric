# SearchFabric

Local multimodal search app with fast hybrid retrieval.

## What Changed

The image search pipeline now uses a RAM++-first architecture:

- Images are indexed with RAM++ tags.
- RAM++ tags are fused into semantic descriptions and embedded with SentenceTransformers.
- Query-time search blends text FTS with semantic retrieval over the same embedding space.

This replaces the previous CLIP-centric image path with tag-aware retrieval that is usually better for keyword/tag search while remaining low-latency locally.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download a RAM++ checkpoint, for example:

- `ram_plus_swin_large_14m.pth`

3. Set the checkpoint path:

```bash
# PowerShell
$env:RAM_PLUS_CHECKPOINT="G:\models\ram_plus_swin_large_14m.pth"

# CMD
set RAM_PLUS_CHECKPOINT=G:\models\ram_plus_swin_large_14m.pth
```

4. Run the app:

```bash
python src/app.py
```

## Search Architecture

- Text/code files: SQLite FTS5 index in `TextIndexer`.
- Images/PDFs: semantic index in `SemanticIndexer`.
- Images:
	- RAM++ generates tags.
	- Tags + filename (+ optional context) are embedded with the selected sentence-transformer model.
- Queries:
	- Hybrid search merges direct text matches and semantic matches.
	- Semantic scores include a small keyword-overlap boost against RAM++ tags.

## Notes

- If `RAM_PLUS_CHECKPOINT` is not configured, image indexing falls back to filename-derived tags.
- For best image search quality, configure RAM++ and re-index files.
