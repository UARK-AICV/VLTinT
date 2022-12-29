VLTinT: Visual-Linguistic Transformer-in-Transformer for Coherent Video Paragraph Captioning
=====
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vltint-visual-linguistic-transformer-in/video-captioning-on-activitynet-captions)](https://paperswithcode.com/sota/video-captioning-on-activitynet-captions?p=vltint-visual-linguistic-transformer-in)

Video paragraph captioning aims to generate a multi-sentence description of an untrimmed video with several temporal event locations in coherent storytelling. 
Following the human perception process, where the scene is effectively understood by decomposing it into visual (e.g. human, animal) and non-visual components (e.g. action, relations) under the mutual influence of vision and language, we first propose a visual-linguistic (VL) feature. In the proposed VL feature, the scene is modeled by three modalities including (i) a global visual environment; (ii) local visual main agents; (iii) linguistic scene elements. We then introduce an autoregressive **Transformer-in-Transformer (TinT)** to simultaneously capture the semantic coherence of intra- and inter-event contents within a video. Finally, we present a new **VL contrastive loss function** to guarantee learnt embedding features are matched with the captions semantics. Comprehensive experiments and extensive ablation studies on ActivityNet Captions and YouCookII datasets show that the proposed Visual-Linguistic Transformer-in-Transform (VLTinT) outperforms prior state-of-the-art methods on accuracy and diversity. 

## AAAI Project Page
To deploy on local machine:

```bash
bundle exec jekyll serve
```


## Citation
If you find this code useful for your research, please cite our papers:

```bibtex
@ARTICLE{2022arXiv221115103Y,
       author = {{Yamazaki}, Kashu and {Vo}, Khoa and {Truong}, Sang and {Raj}, Bhiksha and {Le}, Ngan},
        title = "{VLTinT: Visual-Linguistic Transformer-in-Transformer for Coherent Video Paragraph Captioning}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = nov,
          eid = {arXiv:2211.15103},
        pages = {arXiv:2211.15103},
archivePrefix = {arXiv},
       eprint = {2211.15103},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221115103Y},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

```bibtex
@INPROCEEDINGS{9897766,
  author={Yamazaki, Kashu and Truong, Sang and Vo, Khoa and Kidd, Michael and Rainwater, Chase and Luu, Khoa and Le, Ngan},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)}, 
  title={VLCAP: Vision-Language with Contrastive Learning for Coherent Video Paragraph Captioning}, 
  year={2022},
  volume={},
  number={},
  pages={3656-3661},
  doi={10.1109/ICIP46576.2022.9897766}}
```

## Acknowledgement
We acknowledge the following open-source projects that we based on our work:

1. [nerfies project page](https://github.com/nerfies/nerfies.github.io) 
