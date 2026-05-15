---
created: '2026-03-25T08:07:50+00:00'
evidence:
- stage-03/search_plan.yaml
- stage-03/sources.json
- stage-03/queries.json
id: search_strategy-rc-20260325-080637-b1c38b
run_id: rc-20260325-080637-b1c38b
stage: 03-search_strategy
tags:
- search_strategy
- stage-03
- run-rc-20260
title: 'Stage 03: Search Strategy'
---

# Stage 03: Search Strategy

deduplication:
  fuzzy_threshold: 0.9
  method: title_doi_hash
filters:
  language:
  - en
  min_year: 2020
  peer_review_preferred: true
generated: '2026-03-25T08:07:50+00:00'
search_strategies:
- max_results_per_query: 60
  name: keyword_core
  queries:
  - pairwise grasp scoring BPR open-world robotic grasping
  - pairwise grasp
  - grasp scoring
  - scoring open-world
  - open-world robotic
  sources:
  - arxiv
  - semantic_scholar
  - openreview
- depth: 1
  name: backward_forward_citation
  queries:
  - pairwise grasp scoring BPR open-world robotic grasping
  - pairwise grasp
  - grasp scoring
  sources:
  - semantic_scholar
  - google_scholar
topic: pairwise grasp scoring BPR open-world robotic grasping


{
  "sources": [
    {
      "id": "arxiv",
      "name": "arXiv",
      "type": "api",
      "url": "https://export.arxiv.org/api/query",
      "status": "available",
      "query": "pairwise grasp scoring BPR open-world robotic grasping",
      "verified_at": "2026-03-25T08:07:50+00:00"
    },
    {
      "id": "semantic_scholar",
      "name": "Semantic Scholar",
      "type": "api",
      "url": "https://api.semanticscholar.org/graph/v1/paper/search",
      "status": "available",
      "query": "pairwise grasp scoring BPR open-world robotic grasping",
      "verified_at": "2026-03-25T08:07:50+00:00"
    }
  ],
  "count": 2,
  "generated": "2026-03-25T08:07:50+00:00"
}

{
  "queries": [
    "pairwise grasp scoring BPR open-world robotic grasping",
    "pairwise grasp",
    "grasp scoring",
    "scoring open-world",
    "open-world robotic"
  ],
  "year_min": 2020
}