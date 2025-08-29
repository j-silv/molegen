---
title: Molegen
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Two-Step Graph Convolutional Decoder for Molecule Generation
license: mit
---


# Molecular generation

I started this project to get hands-on experience with generative models and explore ML for drug discovery. 

The code is a replication of the following paper: [A Two-Step Graph Convolutional Decoder for Molecule Generation](https://arxiv.org/abs/1906.03412) by Bresson et Laurent (2019).

The documentation for the project is hosted on https://huggingface.co/spaces/j-silv/molegen. You can navigate there directly, or view the embed below.

<iframe
	id="molegen-frame-1"
	src="https://j-silv-molegen.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
<script src="https://cdn.jsdelivr.net/npm/iframe-resizer@4.3.4/js/iframeResizer.min.js"></script>
<script>
  iFrameResize({}, "#molegen-frame-1")
</script>