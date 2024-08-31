# Copyright (c) 2019 NVIDIA Corporation


The English language has this property that the larger the dataset, the higher the average number of times a word appears in that dataset. Let FREQ denote the average number of times a token appears in a dataset. Low FREQ makes it hard for neural language models to do well on small datasets.

SimpleBooks is a small long-term dependency dataset that has the FREQ number equivalent to the 1 billion token dataset. Its small vocabulary size and small percentage of out-of-vocabulary words make it an ideal testbed and benchmark for word-level language modeling task and tutorials.

It was created from 1,573 Gutenberg books. They were selected out of 39,432 Gutenberg books using a hill-climbing algorithm to maximize FREQ.


SimpleBooks comes in two sizes:
- SimpleBooks-2 (11MB): 2.2M tokens with the vocab size of 11,492.
- SimpleBooks-92 (409MB): 92M tokens with the vocab size of 98,304.

Each size comes with the tokenized version (simplebooks-[2/92]) and a raw version that hasn't been tokenized (simplebooks-[2/92]-raw).

SimpleBooks compared to several other small/medium sized language modeling datasets"

 				Source	 	Tokens	 Vocab		OOV	 	FREQ
1Billion	 	News	 	829M	 793,471	0.28%	1045.09
WikiText-103	Wikipedia	103M	 267,735	0.4%	385.56
WikiText-2		Wikipedia	 2M		 33,278		2.6%	62.76
PTB				News		0.9M	 10,000		4.8%	88.75
==============================================================
SimpleBooks-92	Books		91.5M	 98,304		0.11%	931.4
SimpleBooks-2	Books	 	2.2M	 11,492		0.47%	195.43

SimpleBooks is distributed under Creative Common Attribution CC-BY license.
If you use this dataset, please cite:

`SimpleBooks: Long-term dependency book dataset with simplifiedEnglish vocabulary for word-level language modeling` (Huyen Nguyen, 2019)

The paper is availabe on arxiv!