# Changelog

## 0.1.2 (2026-11-06)

- Tensor Viewer asks one time for a Marketplace rating. It counts each success (an opened tensor file). After 5 of them, and no sooner than 2 days after the first, it shows a small message with "Rate it", "Later" and "Don't ask again". "Later" waits for 20 more, at most two more times. It never shows in your first session, more than once per session, when the window is not focused, or in a remote or untrusted workspace.
- New setting `tensorViewer.askForRating` (default `true`) turns it off. Nothing is sent anywhere; "Rate it" only opens the review page in your browser.

## 0.1.1 (2026-11-06)

- Listing only, no change to the extension. The display name is now "GGUF, Safetensors & NPY Tensor Viewer" and the description is one sentence, so the listing uses the words people search for (gguf (28), npy (16)).
- The 30 keywords start with the most searched terms.
- The README ends with links to the other Zovo extensions.

## 0.1.0

First release. Opens .npy, .npz, .safetensors and .gguf files offline. Tensor table with search and sort, GGUF metadata and tokenizer summary, total parameter count, per-tensor stats, histogram, heatmap, line plot and image view. Decodes F32, F16, BF16, FP8, the integer types, and the GGUF quant types Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K.
