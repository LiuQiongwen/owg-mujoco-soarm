---
created: '2026-03-25T05:42:54+00:00'
evidence:
- stage-03/search_plan.yaml
- stage-03/sources.json
- stage-03/queries.json
id: search_strategy-rc-20260325-054058-b1c38b
run_id: rc-20260325-054058-b1c38b
stage: 03-search_strategy
tags:
- search_strategy
- stage-03
- run-rc-20260
title: 'Stage 03: Search Strategy'
---

# Stage 03: Search Strategy

strategy:
  deduplication: doi_or_title_fuzzy_match
  rounds:
  - focus: BPR pairwise ranking for robotic grasping
    id: R1
    queries:
    - Bradley-Terry pairwise ranking robotic grasp selection
    - pairwise preference learning grasp quality scoring
    - BPR loss grasp candidate ranking point cloud
    sources:
    - S1
    - S2
    - S3
  - focus: open-world grasp generalization and sim-to-real
    id: R2
    queries:
    - open-world robotic grasping novel objects generalization
    - sim-to-real transfer pairwise grasp labels
    - language-conditioned grasp scoring VLM grounding
    sources:
    - S1
    - S2
    - S4
  - focus: grasp candidate generation and ranking efficiency
    id: R3
    queries:
    - grasp candidate set size ranking benefit analysis
    - geometric feature ablation grasp quality prediction
    - antipodal grasp scoring surface normal curvature
    sources:
    - S1
    - S2
    - S3
    - S5
  stopping_criteria: '>=15 unique papers OR 3 rounds completed'
topic: pairwise grasp scoring BPR open-world robotic grasping


{
  "sources": [
    {
      "id": "S1",
      "name": "arXiv (cs.RO, cs.LG)",
      "type": "preprint_server",
      "url": "https://arxiv.org/search/?searchtype=all&query=pairwise+grasp+ranking+robotic&start=0",
      "status": "active",
      "query": "pairwise grasp ranking BPR open-world robotic",
      "verified_at": "2026-03-25"
    },
    {
      "id": "S2",
      "name": "Semantic Scholar",
      "type": "academic_search",
      "url": "https://api.semanticscholar.org/graph/v1/paper/search?query=pairwise+grasp+scoring+open-world+robotic&fields=title,authors,year,abstract,citationCount",
      "status": "active",
      "query": "pairwise grasp scoring open-world robotic grasping BPR",
      "verified_at": "2026-03-25"
    },
    {
      "id": "S3",
      "name": "IEEE Xplore",
      "type": "academic_database",
      "url": "https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=robotic+grasp+ranking+pairwise+learning",
      "status": "active",
      "query": "robotic grasp ranking pairwise learning point cloud",
      "verified_at": "2026-03-25"
    },
    {
      "id": "S4",
      "name": "ACM Digital Library",
      "type": "academic_database",
      "url": "https://dl.acm.org/action/doSearch?query=language+conditioned+grasp+scoring+open+world",
      "status": "active",
      "query": "language conditioned grasp scoring open world VLM",
      "verified_at": "2026-03-25"
    },
    {
      "id": "S5",
      "name": "Papers With Code",
      "type": "benchmark_tracker",
      "url": "https://paperswithcode.com/task/robotic-grasping",
      "status": "active",
      "query": "robotic grasping SOTA pairwise ranking sim-to-real",
      "verified_at": "2026-03-25"
    }
  ],
  "count": 5,
  "generated": "2026-03-25T05:42:54+00:00"
}

{
  "queries": [
    "pairwise grasp scoring BPR open world",
    "pairwise grasp scoring BPR benchmark",
    "pairwise grasp scoring BPR survey",
    "grasp scoring BPR open",
    "pairwise grasp scoring comparison",
    "pairwise grasp scoring deep learning",
    "scoring BPR open world"
  ],
  "year_min": 2020
}