---
created: '2026-03-25T05:43:58+00:00'
evidence:
- stage-04/candidates.jsonl
- stage-04/web_context.md
- stage-04/web_search_result.json
- stage-04/references.bib
- stage-04/search_meta.json
id: literature_collect-rc-20260325-054058-b1c38b
run_id: rc-20260325-054058-b1c38b
stage: 04-literature_collect
tags:
- literature_collect
- stage-04
- run-rc-20260
title: 'Stage 04: Literature Collect'
---

# Stage 04: Literature Collect

{"paper_id": "oalex-W2958089299", "title": "A Survey on Explainable Artificial Intelligence (XAI): Toward Medical XAI", "authors": [{"name": "Erico Tjoa", "affiliation": "Alibaba Group (China)"}, {"name": "Cuntai Guan", "affiliation": "Nanyang Technological University"}], "year": 2020, "abstract": "Recently, artificial intelligence and machine learning in general have demonstrated remarkable performances in many tasks, from image processing to natural language processing, especially with the advent of deep learning (DL). Along with research progress, they have encroached upon many different fields and disciplines. Some of them require high level of accountability and thus transparency, for example, the medical sector. Explanations for machine decisions and predictions are thus needed to justify their reliability. This requires greater interpretability, which often means we need to understand the mechanism underlying the algorithms. Unfortunately, the blackbox nature of the DL is still unresolved, and many machine decisions are still poorly understood. We provide a review on interpretabilities suggested by different research works and categorize them. The different categories show different dimensions in interpretability research, from approaches that provide \"obviously\" interpretable information to the studies of complex patterns. By applying the same categorization to interpretability in medical research, it is hoped that: 1) clinicians and practitioners can subsequently approach these methods with caution; 2) insight into interpretability will be born with more considerations for medical practices; and 3) initiatives to push forward data-based, mathematically grounded, and technically grounded medical education are encouraged.", "venue": "IEEE Transactions on Neural Networks and Learning Systems", "citation_count": 1993, "doi": "10.1109/tnnls.2020.3027314", "arxiv_id": "", "url": "https://doi.org/10.1109/tnnls.2020.3027314", "source": "openalex", "cite_key": "tjoa2020survey", "collected_at": "2026-03-25T05:43:46+00:00"}
{"paper_id": "oalex-W3149839747", "title": "Ensemble deep learning: A review", "authors": [{"name": "M. A. Ganaie", "affiliation": "Indian Institute of Technology Indore"}, {"name": "Minghui Hu", "affiliation": "Nanyang Technological University"}, {"name": "A. K. Malik", "affiliation": "Indian Institute of Technology Indore"}, {"name": "M. Tanveer", "affiliation": "Indian Institute of Technology Indore"}, {"name": "Ponnuthurai Nagaratnam Suganthan", "affiliation": "Qatar University"}], "year": 2022, "abstract": "", "venue": "Engineering Applications of Artificial Intelligence", "citation_count": 1911, "doi": "10.1016/j.engappai.2022.105151", "arxiv_id": "", "url": "https://doi.org/10.1016/j.engappai.2022.105151", "source": "openalex", "cite_key": "ganaie2022ensemble", "collected_at": "2026-03-25T05:43:46+00:00"}
{"paper_id": "oalex-W4362515116", "title": "A Survey of Large Language Models", "authors": [{"name": "Wayne Xin Zhao", "affiliation": ""}, {"name": "Kun Zhou", "affiliation": ""}, {"name": "Junyi Li", "affiliation": ""}, {"name": "Tianyi Tang", "affiliation": ""}, {"name": "Xiaolei Wang", "affiliation": ""}, {"name": "Yupeng Hou", "affiliation": ""}, {"name": "Yingqian Min", "affiliation": ""}, {"name": "Beichen Zhang", "affiliation": ""}, {"name": "Junjie Zhang", "affiliation": ""}, {"name": "Zican Dong", "affiliation": ""}, {"name": "Yifan Du", "affiliation": ""}, {"name": "Yang Chen", "affiliation": ""}, {"name": "Yushuo Chen", "affiliation": ""}, {"name": "Zhipeng Chen", "affiliation": ""}, {"name": "Jinhao Jiang", "affiliation": ""}, {"name": "Ruiyang Ren", "affiliation": ""}, {"name": "Yifan Li", "affiliation": ""}, {"name": "Xinyu Tang", "affiliation": ""}, {"name": "Zikang Liu", "affiliation": ""}, {"name": "Peiyu Liu", "affiliation": ""}, {"name": "Jian‐Yun Nie", "affiliation": ""}, {"name": "Ji-Rong Wen", "affiliation": ""}], "year": 2023, "abstract": "Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable AI algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pre-training Transformer models over large-scale corpora, showing strong capabilities in solving various NLP tasks. Since researchers have found that model scaling can lead to performance improvement, they further study the scaling effect by increasing the model size to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement but also show some special abilities that are not present in small-scale language models. To discriminate the diffe

... (truncated, see full artifact)


## Web Search Results
### [1] 21.5. Personalized Ranking for Recommender Systems - D2L
URL: https://d2l.ai/chapter_recommender-systems/ranking.html
21.5.1. Bayesian Personalized Ranking Loss and its Implementation Bayesian personalized ranking (BPR) (Rendle et al., 2009) is a pairwise personalized ranking loss that is derived from the maximum posterior estimator. It has been widely used in many existing recommendation models. The training data of BPR consists of both positive and negative pairs (missing values). It assumes that the user ...

### [2] BPR_Bayesian-Personalized-Ranking_MPR_Multiple-Pairwise-Ranking
URL: https://github.com/RunlongYu/BPR_MPR
Implement Steffen Rendle, et al. Bayesian personalized ranking from implicit feedback (run BPR.py); Implement Runlong Yu, et al. Multiple Pairwise Ranking with Implicit Feedback (run MPR.py) in Python3.

### [3] Putting BPR in a Box: Bounding the Score Space in Bayesian...
URL: https://openreview.net/forum?id=zHiM0KaGZN
Bayesian Personalized Ranking (BPR) has been widely adopted for recommendation by optimizing pairwise score objectives. However, existing BPR-based methods typically focus on enlarging pairwise score differences, which often leads to excessive separation or clustering of pairwise data—an issue that can result in suboptimal performance.

### [4] Understanding and Implementing BPR Loss in PyTorch
URL: https://www.codegenes.net/blog/bpr-loss-pytorch/
In the field of recommender systems, pairwise ranking losses play a crucial role in training models to rank items according to a user&#x27;s preference. One such widely-used pairwise ranking loss is the Bayesian Personalized Ranking (BPR) loss. PyTorch, a popular deep learning framework, provides the flexibility to implement and optimize models using BPR loss. This blog post aims to provide a ...

### [5] A framework for unbiased explainable pairwise ranking for ...
URL: https://www.sciencedirect.com/science/article/pii/S2665963821000920
Our open-source framework includes code to train and tune state-of-the-art pairwise ranking recommender systems on benchmark datasets and evaluate them based on the three criteria of ranking accuracy, explainability, and popularity debiasing.

### [6] arXiv:2406.18722v4 [cs.RO] 13 Oct 2024
URL: https://arxiv.org/pdf/2406.18722
We propose OWG, an open-world grasping pipeline that combines VLMs with segmentation and grasp synthesis models to unlock grounded world understand- ing in three stages: open-ended referring segmentation, grounded grasp planning and grasp ranking via contact reasoning, all of which can be applied zero-shot via suitable visual prompting mechanisms.

### [7] A framework for unbiased explainable pairwise ranking for recommendation
URL: https://par.nsf.gov/servlets/purl/10354785
Our open-source framework includes code to train and tune state-of-the-art pairwise ranking recommender systems on benchmark datasets and evaluate them based on the three criteria of ranking accuracy, explainability, and popularity debiasing.

### [8] PDF SPR: Similarity pairwise ranking for personalized recommendation
URL: https://dianziliu.github.io/files/spr_kbs22.pdf
Among recommendation strategies, collaborative filtering al-gorithms use the wisdom and behavior of the public to achieve good performance, and thus they have attracted the attention of many researchers [6,7]. Bayesian personalized ranking (BPR) is one such collaborative filtering method, and it is seminal in modeling pairwise learning from the Bayesian perspective [8]. BPR tries to learn from ...

### [9] Learning-to-rank using the WARP loss — LightFM 1.16 documentation - Lyst
URL: https://making.lyst.com/lightfm/docs/examples/warp_loss.html
Learning-to-rank using the WARP loss LightFM is probably the only recommender package implementing the WARP (Weighted Approximate-Rank Pairwise) loss for implicit feedback learning-to-rank. Generally, it perfoms better than the more popular BPR (Bayesian Personalised Ranking) loss — often by a large margin.

### [10] Recommender System using Bayesian Personalized Ranking
URL: https://www.geeksforgeeks.org/machine-learning/recommender-system-using-bayesian-personalized-ranking/
What is Bayesian Personalized Ranking? Bayesian Personalized Ranking is a machine learning algorithm specifically designed for enhancing the recommendation process. It operates under a pairwise ranking framework where the goal is not just to predict the items a user might like but to rank them in the order of potential interest. Unlike traditional methods that might predict absolute ratings ...

### [11] MSBPR: A multi-pairwise preference and similarity based Bayesian ...
URL: https://www.sciencedirect.com/science/article/pii/S0950705122012618
For addressing the &quot;One-Class Collaborative Filtering&quot; (OCCF) problem in recommendation systems, in which the obtained user information is all single-type positive feedback, the current mainstream methods are all based on the idea of pairwise pre

... (truncated, see full artifact)


{
  "topic": "pairwise grasp scoring BPR open-world robotic grasping",
  "web_results_count": 17,
  "scholar_papers_count": 0,
  "crawled_pages_count": 3,
  "pdf_extractions_count": 0,
  "has_search_answer": false,
  "elapsed_seconds": 11.854094435999286,
  "web_results": [
    {
      "title": "21.5. Personalized Ranking for Recommender Systems - D2L",
      "url": "https://d2l.ai/chapter_recommender-systems/ranking.html",
      "snippet": "21.5.1. Bayesian Personalized Ranking Loss and its Implementation Bayesian personalized ranking (BPR) (Rendle et al., 2009) is a pairwise personalized ranking loss that is derived from the maximum posterior estimator. It has been widely used in many existing recommendation models. The training data of BPR consists of both positive and negative pairs (missing values). It assumes that the user ...",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "BPR_Bayesian-Personalized-Ranking_MPR_Multiple-Pairwise-Ranking",
      "url": "https://github.com/RunlongYu/BPR_MPR",
      "snippet": "Implement Steffen Rendle, et al. Bayesian personalized ranking from implicit feedback (run BPR.py); Implement Runlong Yu, et al. Multiple Pairwise Ranking with Implicit Feedback (run MPR.py) in Python3.",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "Putting BPR in a Box: Bounding the Score Space in Bayesian...",
      "url": "https://openreview.net/forum?id=zHiM0KaGZN",
      "snippet": "Bayesian Personalized Ranking (BPR) has been widely adopted for recommendation by optimizing pairwise score objectives. However, existing BPR-based methods typically focus on enlarging pairwise score differences, which often leads to excessive separation or clustering of pairwise data\u2014an issue that can result in suboptimal performance.",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "Understanding and Implementing BPR Loss in PyTorch",
      "url": "https://www.codegenes.net/blog/bpr-loss-pytorch/",
      "snippet": "In the field of recommender systems, pairwise ranking losses play a crucial role in training models to rank items according to a user&#x27;s preference. One such widely-used pairwise ranking loss is the Bayesian Personalized Ranking (BPR) loss. PyTorch, a popular deep learning framework, provides the flexibility to implement and optimize models using BPR loss. This blog post aims to provide a ...",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "A framework for unbiased explainable pairwise ranking for ...",
      "url": "https://www.sciencedirect.com/science/article/pii/S2665963821000920",
      "snippet": "Our open-source framework includes code to train and tune state-of-the-art pairwise ranking recommender systems on benchmark datasets and evaluate them based on the three criteria of ranking accuracy, explainability, and popularity debiasing.",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "arXiv:2406.18722v4 [cs.RO] 13 Oct 2024",
      "url": "https://arxiv.org/pdf/2406.18722",
      "snippet": "We propose OWG, an open-world grasping pipeline that combines VLMs with segmentation and grasp synthesis models to unlock grounded world understand- ing in three stages: open-ended referring segmentation, grounded grasp planning and grasp ranking via contact reasoning, all of which can be applied zero-shot via suitable visual prompting mechanisms.",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "A framework for unbiased explainable pairwise ranking for recommendation",
      "url": "https://par.nsf.gov/servlets/purl/10354785",
      "snippet": "Our open-source framework includes code to train and tune state-of-the-art pairwise ranking recommender systems on benchmark datasets and evaluate them based on the three criteria of ranking accuracy, explainability, and popularity debiasing.",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "PDF SPR: Similarity pairwise ranking for personalized recommendation",
      "url": "https://dianziliu.github.io/files/spr_kbs22.pdf",
      "snippet": "Among recommendation strategies, collaborative filtering al-gorithms use the wisdom and behavior of the public to achieve good performance, and thus they have attracted the attention of many researchers [6,7]. Bayesian personalized ranking (BPR) is one such collaborative filtering method, and it is seminal in modeling pairwise learning from the Bayesian perspective [8]. BPR tries to learn from ...",
      "content": "",
      "score": 0.0,
      "source": "duckduckgo"
    },
    {
      "title": "Learning-to-rank using the WARP loss \u2014 LightFM 1.16 documentation - Lyst",
      "url": "https://making.lyst.com/lightfm/docs/examples/warp_loss.html",
      "snippet

... (truncated, see full artifact)


@article{tjoa2020survey,
  title = {A Survey on Explainable Artificial Intelligence (XAI): Toward Medical XAI},
  author = {Erico Tjoa and Cuntai Guan},
  year = {2020},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  doi = {10.1109/tnnls.2020.3027314},
  url = {https://doi.org/10.1109/tnnls.2020.3027314},
}

@article{ganaie2022ensemble,
  title = {Ensemble deep learning: A review},
  author = {M. A. Ganaie and Minghui Hu and A. K. Malik and M. Tanveer and Ponnuthurai Nagaratnam Suganthan},
  year = {2022},
  journal = {Engineering Applications of Artificial Intelligence},
  doi = {10.1016/j.engappai.2022.105151},
  url = {https://doi.org/10.1016/j.engappai.2022.105151},
}

@article{zhao2023survey,
  title = {A Survey of Large Language Models},
  author = {Wayne Xin Zhao and Kun Zhou and Junyi Li and Tianyi Tang and Xiaolei Wang and Yupeng Hou and Yingqian Min and Beichen Zhang and Junjie Zhang and Zican Dong and Yifan Du and Yang Chen and Yushuo Chen and Zhipeng Chen and Jinhao Jiang and Ruiyang Ren and Yifan Li and Xinyu Tang and Zikang Liu and Peiyu Liu and Jian‐Yun Nie and Ji-Rong Wen},
  year = {2023},
  journal = {ArXiv.org},
  doi = {10.48550/arxiv.2303.18223},
  url = {https://doi.org/10.48550/arxiv.2303.18223},
}

@article{park2023generative,
  title = {Generative Agents: Interactive Simulacra of Human Behavior},
  author = {Joon Sung Park and Joseph O’Brien and Carrie J. Cai and Meredith Ringel Morris and Percy Liang and Michael S. Bernstein},
  year = {2023},
  doi = {10.1145/3586183.3606763},
  url = {https://doi.org/10.1145/3586183.3606763},
}

@article{burkart2021survey,
  title = {A Survey on the Explainability of Supervised Machine Learning},
  author = {Nadia Burkart and Marco F. Huber},
  year = {2021},
  journal = {Journal of Artificial Intelligence Research},
  doi = {10.1613/jair.1.12228},
  url = {https://doi.org/10.1613/jair.1.12228},
}

@article{plompen2020joint,
  title = {The joint evaluated fission and fusion nuclear data library, JEFF-3.3},
  author = {Arjan Plompen and Ó. Cabellos and C. De Saint Jean and Michael Fleming and A. Algora and M. Angelone and P. Archier and E. Bauge and O. Bersillon and A. I. Blokhin and F. Cantargi and A. Chebboubi and C.J. Díez and H. Duarte and E. Dupont and James Dyrda and Bernard Erasmus and Luca Fiorito and U. Fischer and D. Flammini and Daniela Foligno and Mark R. Gilbert and J.R. Granada and Wim Haeck and F.-J. Hambsch and Petter Helgesson and S. Hilaire and I. D. Hill and Mathieu Hursin and R. Ichou and R. Jacqmin and Bohumil Jánský and Cédric Jouanne and M.A. Kellett and D. H. Kim and H. I. Kim and I. Kodeli and A. J. Koning and A.Yu. Konobeyev and S. Kopecky and Bor Kos and A. Krása and L.C. Leal and Nicolas Leclaire and Philippe Le Conte and Young-Chan Lee and H. Leeb and O. Litaize and M. Majerle and J. I Márquez Damián and F. Michel-Sendis and R.W. Mills and Benjamin Morillon and G. Noguère and M. Pecchia and S. Pelloni and P. Pereslavtsev and Robert J. Perry and D. Rochman and A. Röhrmoser and P. Romain and Pablo Romojaro and D. Roubtsov and P. Sauvan and P. Schillebeeckx and Konrad Schmidt and O. Sérot and Sergey Simakov and I. Sirakov and Henrik Sjöstrand and Alexey Stankovskiy and Jean-Christophe Sublet and Pierre Tamagno and Andrej Trkov and S.C. van der Marck and F. Álvarez‐Velarde and R. Villari and Thomas Ware and Keiichi Yokoyama and Gašper Žerovnik},
  year = {2020},
  journal = {The European Physical Journal A},
  doi = {10.1140/epja/s10050-020-00141-9},
  url = {https://doi.org/10.1140/epja/s10050-020-00141-9},
}

@article{gao2023survey,
  title = {A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions},
  author = {Chen Gao and Yu Zheng and Nian Li and Yinfeng Li and Yingrong Qin and Jinghua Piao and Yuhan Quan and Jianxin Chang and Depeng Jin and Xiangnan He and Yong Li},
  year = {2023},
  journal = {ACM Transactions on Recommender Systems},
  doi = {10.1145/3568022},
  url = {https://doi.org/10.1145/3568022},
}

@article{meteyard2020best,
  title = {Best practice guidance for linear mixed-effects models in psychological science},
  author = {Lotte Meteyard and Robert Davies},
  year = {2020},
  journal = {Journal of Memory and Language},
  doi = {10.1016/j.jml.2020.104092},
  url = {https://doi.org/10.1016/j.jml.2020.104092},
}

@article{zhang2020explainable,
  title = {Explainable Recommendation: A Survey and New Perspectives},
  author = {Yongfeng Zhang and Xu Chen},
  year = {2020},
  journal = {Foundations and Trends® in Information Retrieval},
  doi = {10.1561/1500000066},
  url = {https://doi.org/10.1561/1500000066},
}

@article{gloster2020empirical,
  title = {The empirical status of acceptance and commitment therapy: A review of meta-analyses},
  author = {Andrew T. Gloster and Noemi Walder and Michael E. Levin and Michael P. Twohig and Maria Karekla},
  year = {2020},
  journal = {Journal of Contextual Behavioral Science},
  doi = {10.1016/j.jcbs

... (truncated, see full artifact)


{
  "real_search": true,
  "queries_used": [
    "pairwise grasp scoring BPR open world",
    "pairwise grasp scoring BPR benchmark",
    "pairwise grasp scoring BPR survey",
    "grasp scoring BPR open",
    "pairwise grasp scoring comparison",
    "pairwise grasp scoring deep learning",
    "scoring BPR open world"
  ],
  "year_min": 2020,
  "total_candidates": 180,
  "bibtex_entries": 177,
  "ts": "2026-03-25T05:43:58+00:00"
}