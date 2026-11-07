# Changelog

## 0.3.2 (2026-11-08)

- The "More free viewers" list and the README now use the current Marketplace names and include the newest Zovo extensions: ZIP, GCode, EPUB, HAR, font and audio viewers.

## 0.3.1 (2026-11-07)

- The Marketplace name is now "NPZ Viewer & NPY Viewer, GGUF & Safetensors (NumPy Viewer)" (it was "NPZ & NPY Viewer, GGUF & Safetensors (NumPy Viewer)"). The extension id, commands and settings stay the same.

## 0.3.0 (2026-11-07)

- Arrays as pictures. Any array shaped `(H, W)`, `(H, W, C)` or `(C, H, W)` with 1, 3 or 4 channels, with any number of leading frame dimensions, shows in an Image section. You can pick the layout, the frame and the value range. Only the rows the picture needs are read. Before, only `(H, W, 3)` uint8 arrays were drawn.
- Raw binary arrays. "Open as Raw Array..." (Explorer right-click on `.bin`, `.raw`, `.dat` and similar, or the Command Palette) reads files from `ndarray.tofile()` with the dtype, shape, header size and byte order you give. Stats, values, pictures and Save as .npy work as for `.npy`. The layout is kept per file, and "Change type and shape..." changes it.
- An empty file now shows "is empty (0 bytes)" instead of a parse error.
- Marketplace name is now "NPZ & NPY Viewer, GGUF & Safetensors (NumPy Viewer)", so a search for NPZ viewer finds it. The extension ID, settings and commands stay the same.

## 0.2.4 (2026-11-07)

- "More free viewers" now starts with the Zovo extensions most related to this one. Picking one opens its Marketplace page, where you choose to install it.
- The README has a "Support Zovo" section with GitHub Sponsors and the developer's GitHub profile, and "More tools by Zovo" lists related extensions.

## 0.2.3 (2026-11-07)

- The footer under the viewer has a "Rate" link next to Sponsor and More free viewers. It opens the review page of this extension on the Marketplace. Nothing pops up and nothing asks you for a rating. "Hide" still hides the whole line.
- Marketplace search keywords updated.

## 0.2.2 (2026-11-07)

- The "More free viewers" list and the README list now show the 15 Zovo extensions on the Marketplace under their current names, and 12 that are coming soon.
- Extensions for related files now come first, starting with HDF5 & NetCDF Explorer (H5 Viewer) and CSV Viewer, Excel Viewer & Parquet Viewer.

## 0.2.1 (2026-11-07)

- The Marketplace name is now "GGUF, Safetensors & NPY Viewer (NumPy Viewer)", so a search for NumPy viewer finds it. The extension ID, settings and commands stay the same, and "tensor viewer" stays in the tags.

## 0.2.0 (2026-11-06)

- New Values table under the heatmap. It shows the numbers of any tensor, 32 rows at a time, read by offset so big files stay fast. Page with the arrow buttons or type a row or column number. Pick Auto, Exact, 2, 4 or 8 decimals, or Scientific. Hover a cell for its index and exact value. "Copy values" copies the page as tab-separated text.
- GGUF quant values are shown after dequantizing. float16, bfloat16 and FP8 show their exact float32 value. Complex numbers show as `3+4j`.
- Tensors with 3 or more dimensions get a Slice field, such as `[5, :, :]`. A tick box makes the stats, the heatmap and Save use that slice too.
- New "Save as .npy..." and "Save as CSV..." buttons write a tensor or a slice to a file you pick. Types NumPy knows are copied byte for byte, and bfloat16, FP8 and GGUF quants are written as float32. The file is written in chunks with a progress bar you can cancel. Tensor Viewer refuses to write over the file it is showing.
- "Opened in" no longer grows when you come back to a tab that was in the background.

## 0.1.4 (2026-11-06)

- The Marketplace page now starts with a screenshot of a real VS Code window with a GGUF model and a tensor heatmap, and a short GIF of filtering and opening tensors. Both were made from generated sample files. The images load from the GitHub repo and are not packed into the extension, so the download size stays the same.

## 0.1.3 (2026-11-06)

- A small line at the bottom right of the viewer has two links, "Sponsor" (GitHub Sponsors) and "More free viewers". The new setting `tensorViewer.showSupportLinks` (default `true`) hides it, and so does the "Hide" link next to them. There is no popup, and nothing is sent anywhere.
- New commands "Tensor Viewer: Support the developer" and "Tensor Viewer: More free viewers". The second one lists the other Zovo extensions and installs only the one you pick. Ones that are not on the Marketplace yet open their GitHub page.
- The listing gets a Sponsor button. The README has a short Support section and lists all Zovo extensions.

## 0.1.2 (2026-11-06)

- Tensor Viewer asks one time for a Marketplace rating. It counts each success (an opened tensor file). After 5 of them, and no sooner than 2 days after the first, it shows a small message with "Rate it", "Later" and "Don't ask again". "Later" waits for 20 more, at most two more times. It never shows in your first session, more than once per session, when the window is not focused, or in a remote or untrusted workspace.
- New setting `tensorViewer.askForRating` (default `true`) turns it off. Nothing is sent anywhere; "Rate it" only opens the review page in your browser.

## 0.1.1 (2026-11-06)

- Listing only, no change to the extension. The display name is now "GGUF, Safetensors & NPY Tensor Viewer" and the description is one sentence, so the listing uses the words people search for (gguf (28), npy (16)).
- The 30 keywords start with the most searched terms.
- The README ends with links to the other Zovo extensions.

## 0.1.0

First release. Opens .npy, .npz, .safetensors and .gguf files offline. Tensor table with search and sort, GGUF metadata and tokenizer summary, total parameter count, per-tensor stats, histogram, heatmap, line plot and image view. Decodes F32, F16, BF16, FP8, the integer types, and the GGUF quant types Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K.
