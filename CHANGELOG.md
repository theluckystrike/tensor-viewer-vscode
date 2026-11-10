# Changelog

## 0.6.1 (2026-11-10)

- "More free viewers" now starts with the Zovo extensions most related to this one. Picking one opens its Marketplace page, where you choose to install it.
- The README has a "Support Zovo" section with GitHub Sponsors and the developer's GitHub profile, and "More tools by Zovo" lists related extensions.
- The walkthrough ends with a step that links free Zovo extensions for the files that often come with this one. A link opens the extension page; nothing installs on its own.
- The footer reads "Rate · More free viewers · Zovo Pro, USD 19 · PayPal · Hide". "Zovo Pro, USD 19" and the Marketplace Sponsor button open the Zovo Pro tier; "PayPal" opens the PayPal steps with the amount.
- Zovo Pro buy path: the Pro message names the other Pro tools in NumPy Viewer and the number of Zovo extensions Pro covers, with 4 buttons: Get Zovo Pro (USD 19), Pay with PayPal, I already paid, Cancel. "I already paid" turns Pro on with GitHub Sponsors or a license key.
- Turn Zovo Pro on once per computer: the other Zovo extensions you have pick it up from `~/.zovo/pro.json`, with no second sign-in or paste. After an unlock you get one offer to turn it on in your other Zovo tools; "Not now" is final.
- "Support the developer" lists Zovo Pro first. Every Pro link opens the USD 19 Zovo Pro tier on GitHub Sponsors, not the general Sponsors page.
- The 5 Zovo Pro commands share one "Zovo Pro" group in the Command Palette, with the same names in every Zovo extension.
- Nothing that was free changed. No popup on open.

## 0.6.0 (2026-11-09)

Everything that was free stays free. New, with the optional Zovo Pro (USD 19 once for every Zovo viewer; each tool has 3 free tries on your computer):

- Compare two checkpoints: every tensor of two files, value by value. Max and mean absolute difference, RMSE, relative L2 error, cosine similarity, the count of values that differ and where the largest one is. Tensors in one file only, shape and type changes are listed. Works across formats, on quantized GGUF tensors and on Fortran-order arrays. Names also match without a `module.` or `_orig_mod.` prefix.
- Scan all tensors: every value of every tensor, no sampling. Flags NaN, Inf, all-zero and constant tensors and float values of 10,000 or more, with exact min, max, mean, std and L2 norm.
- Both open a report you can filter and save as CSV, Markdown or JSON, and both can be cancelled.
- "Compare with..." and "Scan all tensors" buttons at the top of the viewer, Command Palette commands, and Explorer menu entries. `tensorViewer.showProFeatures: false` hides them.
- Scan all tensors on a file with no readable tensor (for example a pickle the viewer will not run) says so and does not use a free try.
- A file with no tensor (for example a pickle that only calls os.system) no longer counts toward the rating question.
- Buying Zovo Pro: the README lists the PayPal steps (USD 19, or USD 59 for a team of 5) and the transaction ID form.

Also:

- The rating question comes after 2 files that opened without an error in 2 sessions at least 4 hours apart (was 3 files on 2 calendar days). It still never shows after an error, at most twice in all, and the setting turns it off.
- "Rate it" and the Rate link in the footer open the review page of the store your editor installs from: the Visual Studio Marketplace in VS Code, Open VSX in Cursor, Windsurf, VSCodium and other editors that use Open VSX.
- Rate is now the first footer link.

## 0.4.1 (2026-11-09)

- PayPal is now in the footer links and in the README "Support Zovo" section, next to GitHub Sponsors. Both are optional; the viewer works the same without them.
- README: a short "Free and Zovo Pro" note. This viewer stays free and complete; nothing in it becomes paid.
- No change to how files open or look.

## 0.4.0 (2026-11-08)

- Opens PyTorch files: `.pt`, `.pth`, `.ckpt` and `pytorch_model*.bin`. Every tensor is listed by its full name with dtype and shape, and a click shows stats, a heatmap and the values. This covers state dicts, training checkpoints with optimizer state, Lightning checkpoints, whole saved models, TorchScript files and the format from before PyTorch 1.6. Epoch, step and loss show in the header.
- The pickle in a PyTorch file is read without running it, with an allow-list of tensor rebuilders and plain containers. Other Python names are rated in the Metadata tab, risky ones (such as `os.system`) show a warning, and the header says whether `torch.load(weights_only=True)` should load the file.
- A plain-text Python `.pth` path file (the kind in site-packages) opens in the text editor, as before. Only files that are not PyTorch files and read as text take this path.
- Files that share one value many times, or contain themselves, open at once. A crafted file could keep the viewer busy for minutes before.
- More names are rated risky, such as `_posixsubprocess`, `pydoc`, `io.open`, `torch.jit.load` and `torch.utils.cpp_extension`.
- Auto-reload: the view reads the file again when it changes on disk and keeps the selected tensor, filter and tab. Setting: `tensorViewer.autoReload`.
- Marketplace name is now "NPZ Viewer & NPY Viewer, PyTorch, GGUF & Safetensors (NumPy Viewer)". The extension id, commands and settings stay the same.
- The rating question now comes after 3 files that opened without an error on 2 different days, and never after an error. "Later" asks one more time after 7 days, "No thanks" ends it, and it never asks more than 2 times. `tensorViewer.askForRating: false` turns it off.

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
