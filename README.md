# GGUF, Safetensors & NPY Tensor Viewer

[![VS Code Marketplace installs](https://img.shields.io/visual-studio-marketplace/i/zovo.tensor-viewer?label=VS%20Code%20installs)](https://marketplace.visualstudio.com/items?itemName=zovo.tensor-viewer)
[![Open VSX version](https://img.shields.io/open-vsx/v/zovo/tensor-viewer?label=Open%20VSX)](https://open-vsx.org/extension/zovo/tensor-viewer)

For people who run or fine-tune local models, Tensor Viewer opens `.gguf`, `.safetensors`, `.npy` and `.npz` files in VS Code and shows every tensor's name, dtype and shape, plus stats and a heatmap, without Python.

![A llama-style GGUF model in VS Code. The header shows 4.00M parameters, architecture llama and context length 2048. The tensor table lists Q8_0, Q4_0 and F32 tensors, and blk.2.attn_q.weight is open with its stats and a blue and red heatmap](https://raw.githubusercontent.com/theluckystrike/tensor-viewer-vscode/main/images/vscode-gguf.png)

*A real VS Code 1.142 window running this extension on [`test/fixtures/tiny-llama-q8.gguf`](https://github.com/theluckystrike/tensor-viewer-vscode/blob/main/test/fixtures/tiny-llama-q8.gguf), a 3.35 MB llama-style GGUF written by gguf-py from `scripts/make-fixtures.py`.*

- It opens big checkpoints in well under a second and shows the real numbers. Only the header is read at first, and a click on a tensor reads that tensor's bytes by offset. Our test opens a 10.7 GB safetensors file in under 1 s. The Values table pages through any tensor, one 2-D slice at a time, and Save writes a tensor or a slice to .npy or CSV.
- It reads the formats local LLM users actually have. GGUF metadata (architecture, context length, file type, tokenizer, chat template) sits next to the quant type of each tensor. Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K weights are decoded for stats and heatmaps.
- It's safe to open files you downloaded. Nothing in the file is ever run. Pickled object arrays in `.npy` and `.npz` are listed and refused, and the viewer makes no network calls at all.

## Quick start

1. Install Tensor Viewer.
2. Click any `.gguf`, `.safetensors`, `.npy` or `.npz` file in the Explorer. Then click a tensor in the table.

![A click on the GGUF file opens its header and tensor table. Typing attn_q filters the table, and clicks on blk.0.attn_q.weight and blk.2.attn_q.weight show their stats and heatmaps](https://raw.githubusercontent.com/theluckystrike/tensor-viewer-vscode/main/images/demo-tensor.gif)

*Recorded in the same window. The white arrow is drawn in to show where each click lands.*

Run "Tensor Viewer: Get Started" from the Command Palette for a short tour.

## What you see

The top bar shows the total parameter count, the number of tensors, the file size and how long the header took. GGUF files add a summary row with the architecture, model name, context length, embedding size, layers, heads, file type, tokenizer size and bits per weight.

The tensor table lists name, type, shape, parameter count, size in bytes and byte offset. Type in the filter box to match names (`attn_q`, `blk.3.`) or an exact type (`Q4_K`, `float16`). Click a header to sort. Up and down arrows move through the list. Tables with thousands of tensors show 500 rows at a time.

Click a tensor and the right panel shows min, max, mean, std, NaN, +Inf, -Inf and zero counts. 2-D tensors get a heatmap. 1-D tensors get a line plot with a min and max band, and an HxWx3 uint8 array shows as a picture. A histogram comes last. Hover over any of them to read exact values. "Copy shape" puts `(4096, 11008)` on the clipboard, and "Copy torch.Size" gives `torch.Size([4096, 11008])`.

### Values table

Under the heatmap, the Values table shows the numbers themselves, 32 rows at a time. Use the arrow buttons to page through rows and columns, or type a row or column number and press Enter. The number format can be Auto, Exact, 2, 4 or 8 decimals, or Scientific. Hover over a cell to see its index and its exact value. "Copy values" puts the page on the clipboard as tab-separated text, which pastes into a spreadsheet.

GGUF quant types such as Q4_K and Q8_0 show their values after dequantizing, as llama.cpp computes them. float16, bfloat16 and FP8 values show as their exact float32 value. Complex numbers show as `3+4j`.

For tensors with 3 or more dimensions, the Slice field picks one 2-D slice, for example `[5, :, :]` for the sixth image in a batch. The table shows that slice. Tick "Use only this slice for the stats, the heatmap and Save" and the stats, the heatmap and the Save buttons use it too.

### Save as .npy or CSV

"Save as .npy..." writes the tensor (or the slice) to a NumPy file that `np.load` reads. Types NumPy has, such as float32, int64 or complex64, are copied byte for byte. bfloat16, FP8 and the GGUF quant types are written as float32. "Save as CSV..." writes one row per line with every digit kept. 3-D and bigger tensors become rows of their last dimension. You pick the file in a save dialog, the file is written in chunks with a progress bar you can cancel, and Tensor Viewer never writes over the file it is showing.

![A GPT-2 style safetensors file with the wpe.weight positional embedding open. The heatmap shows the sine and cosine wave pattern](https://github.com/theluckystrike/tensor-viewer-vscode/raw/HEAD/images/safetensors.png)

The heatmap uses blue for negative values and red for positive ones, with gray at zero. Data that's never negative uses a single blue ramp instead. Colors are clipped at the 99.5th percentile so a few outlier weights don't flatten everything else, and the caption says when that happened.

![A compressed npz file with a 5,000-value signal shown as a line plot and a histogram](https://github.com/theluckystrike/tensor-viewer-vscode/raw/HEAD/images/npz-signal.png)

![An HxWx3 uint8 numpy array drawn as an image](https://github.com/theluckystrike/tensor-viewer-vscode/raw/HEAD/images/npy-image.png)

## Supported files

| Format | What is read | Types decoded for stats |
|---|---|---|
| `.npy` | header v1, v2 and v3, C and Fortran order, either byte order | bool, int8 to int64, uint8 to uint64, float16/32/64, complex (as magnitude), datetime as raw ticks |
| `.npz` | zip directory, stored and deflated entries, zip64 | same as `.npy` |
| `.safetensors` | JSON header and `__metadata__` | BOOL, U8, I8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64, F8_E4M3, F8_E5M2 |
| `.gguf` | v1 to v3 metadata and tensor info, through @huggingface/gguf | F32, F16, BF16, F64, I8 to I64, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K |

Other GGUF types (Q2_K, Q3_K, IQ types, TQ, MXFP4) are listed with the right size and offset, but have no stats yet. Text, structured and object arrays in NumPy files are listed without stats.

The GGUF shapes are shown in row-major order, the way PyTorch prints them. GGUF itself stores the dimensions fastest-first, so `[4096, 32000]` in llama.cpp logs is `[32000, 4096]` here.

## How big files stay fast

Tensor Viewer never reads a whole file. The `.npy` and safetensors headers are a few hundred bytes to a few MB, and GGUF metadata is read in 2 MB steps until the tensor list ends. For one tensor it reads at most `tensorViewer.sampleValues` values (2,000,000 by default). When a tensor is bigger, it reads up to 512 runs of whole rows spread evenly from the first row to the last, so the heatmap covers the full tensor. The panel always says whether the stats are exact or from a sample, and how many values went in.

Deflated `.npz` entries can't be read at an offset. For those, the header comes from inflating just the start of the entry, and big arrays are sampled from their first rows.

| Setting | Default | What it does |
|---|---|---|
| `tensorViewer.sampleValues` | 2,000,000 | Most values read from one tensor for stats and plots |
| `tensorViewer.maxRemoteFileMB` | 512 | Files not on local disk are read whole, up to this size |
| `tensorViewer.askForRating` | true | Ask one time for a Marketplace rating, after 5 files and 2 days. Turn it off with `false` |

The Values table reads only the cells on the page, so a page of a 10 GB tensor costs a few KB.

## Privacy and safety

There's no telemetry and no network access. The webview runs under a strict Content-Security-Policy with no remote scripts, and every name and metadata string is drawn as text, never as HTML. Python pickle is never loaded, so `.pt`, `.pth`, `.pkl` and object arrays inside `.npy` files stay closed. Tensor Viewer never changes the file it shows. Save as .npy and Save as CSV write only to a new file that you pick.

## Limits

GGUF files split into shards show the tensors of the open shard only, with a note. Heatmaps show at most 256 by 256 cells, each the mean of its block of values. Tensors with three or more dimensions are drawn with the leading dimensions flattened into rows, unless you tick "Use only this slice". Slices of Fortran-order arrays show in the Values table, but the stats and heatmap cover the whole array. CSV export stops at 20 million values; use .npy for bigger tensors. Compressed `.npz` entries can only be read from the start, so values past their first 64 MB are not shown, and entries over 512 MB can't be saved. 64-bit integers above 2^53 lose their last digits in the table and in CSV. The .npy export copies them exactly.

## License

MIT. Bundled libraries are listed in THIRD-PARTY-NOTICES.md (@huggingface/gguf, @huggingface/tasks and fflate, all MIT).

## Support

Zovo extensions have no ads and no tracking, and sponsors pay for the time that goes into fixes and new viewers. You can sponsor the work on [GitHub Sponsors](https://github.com/sponsors/theluckystrike).

## More viewers by Zovo

- [Context Drift](https://marketplace.visualstudio.com/items?itemName=zovo.context-drift) finds stale facts in AGENTS.md, CLAUDE.md and other agent rule files.
- [PDF Viewer & Reader with Dark Mode](https://marketplace.visualstudio.com/items?itemName=zovo.pdf-viewer-dark) reads PDFs with a real dark mode, search, an outline and thumbnails.
- [CSV, Excel & Parquet Viewer](https://marketplace.visualstudio.com/items?itemName=zovo.table-viewer-vscode) opens Parquet, Excel and CSV files as a sortable, filterable table.
- [DOCX Viewer & Word Viewer](https://marketplace.visualstudio.com/items?itemName=zovo.docx-viewer-vscode) previews Word files with dark mode, an outline and search.
- [HDF5 & NetCDF Explorer](https://marketplace.visualstudio.com/items?itemName=zovo.hdf5-netcdf-explorer) browses groups, variables and attributes in scientific data files.
- [PPTX Viewer for PowerPoint](https://marketplace.visualstudio.com/items?itemName=zovo.pptx-slide-viewer) shows PowerPoint slides, thumbnails and speaker notes.
- [Skill Shelf](https://marketplace.visualstudio.com/items?itemName=zovo.skill-shelf) lists, checks and toggles the Agent Skills that Claude Code, Codex and Cursor load.

Coming soon to the Marketplace, with the code on GitHub now.

- [JSONL Viewer](https://github.com/theluckystrike/jsonl-viewer-vscode) opens JSONL and NDJSON files as a table, as records or as a chat view.
- [SQLite Table Viewer](https://github.com/theluckystrike/sqlite-table-viewer) browses SQLite tables read-only and runs SELECT queries.
- [Agent Session Viewer](https://github.com/theluckystrike/agent-session-viewer) browses and searches your local Claude Code and Codex sessions.
- [Audio Preview & Loudness Meter](https://github.com/theluckystrike/audio-preview-vscode) plays audio files with a waveform, a spectrogram and LUFS loudness.
- [Video Preview with Sound](https://github.com/theluckystrike/video-preview-vscode) plays MP4, MOV, MKV and WebM files with their sound.
- [Graphviz Preview & DOT Viewer](https://github.com/theluckystrike/graphviz-preview-vscode) shows a live preview of DOT files and exports SVG or PNG.
- [TIFF Viewer](https://github.com/theluckystrike/tiff-viewer-vscode) opens 16-bit, float, multi-page and GeoTIFF images.
- [Avro, Arrow & Feather Viewer](https://github.com/theluckystrike/arrow-avro-viewer-vscode) opens Avro, Arrow and Feather files as a paged table.
- [EML & MBOX Email Viewer](https://github.com/theluckystrike/eml-viewer-vscode) reads email files with remote images blocked.
- [3D Model Viewer](https://github.com/theluckystrike/model-viewer-vscode) views GLB, glTF, STL, OBJ, PLY and 3MF models.
- [File Viewers Pack](https://github.com/theluckystrike/zovo-file-viewers) installs the PDF, table, Word, PowerPoint, HDF5 and tensor viewers in one step.
