# Sample size

`tensorViewer.sampleValues` (default 2,000,000) caps how many values are read from one tensor. A bigger tensor is sampled by whole rows spread evenly from top to bottom, and the panel says so under the stats.

`tensorViewer.maxRemoteFileMB` (default 512) applies to files that are not on local disk. Those are read whole.
