# A Joint Many-Task Model (JMT)
An implementation of the JMT model proposed in our EMNLP 2017 paper [1]

## Usage
First download the pre-trained word and character n-gram embeddings used in our paper:<br>
`./download_embeddings.sh`<br><br>

Then we can run experiments:<br>
`python train.py`<br>

## Notes
* Currently, only the single-task tagging model is implemented, and eventually all of the five task models will be availabel here.<br>
We can replicate almost the same POS tagging results reported in our paper. We should also be able to replicate the chunking results, but the F1 evaluation metric has not yet implemented.<br>

## Reference ##
[1] <b>Kazuma Hashimoto</b>, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. 2017. A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks. In <i>Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017)</i>, <a href="https://arxiv.org/abs/1611.01587">arXiv cs.CL 1611.01587</a>.

    @InProceedings{hashimoto-jmt:2017:EMNLP2017,
      author    = {Hashimoto, Kazuma and Xiong, Caiming and Tsuruoka, Yoshimasa and Socher, Richard},
      title     = {{A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks}},
      booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      month     = {September},
      year      = {2017},
      address   = {Copenhagen, Denmark},
      publisher = {Association for Computational Linguistics},
      pages      = {446--456},
      url       = {http://arxiv.org/abs/1611.01587}
      }