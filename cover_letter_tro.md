[Date]

Editor-in-Chief
IEEE Transactions on Robotics

Dear Editor-in-Chief,

We would like to submit our manuscript, "Causal-Validity Auditing for Learned Grasp-Candidate Scoring,
with a Case Study in Cross-Embodiment Grasping," for consideration as a regular paper in IEEE Transactions
on Robotics.

This paper is not being submitted to a Special Issue.

**Summary of contribution.** Learned grasp-candidate scorers are only valid if every feature they consume
is computable before the chosen candidate is executed -- a constraint we formalize as causal validity, and
show is easy to violate silently, since an offline accuracy evaluation looks identical whether a feature is
causally valid or not. We build a reference audit tool and demonstrate its practical necessity on our own
cross-embodiment grasp-reranking pipeline: a pooling effect that replicated at p<0.0001 across five
independent pilots collapses to an exact null once causally invalid features are removed. We then automate
the audit with a static dataflow analyzer that independently catches a contamination bug our own hand-built
audit had missed, apply the criterion prospectively to a new investigation from the start (a rebuilt,
causally-admissible-by-construction critic that beats a geometric baseline live-executed on two disjoint
held-out batches, replicating on a fourth object added after the design was frozen while showing a clean
negative on a zero-shot fifth), and report two further independently discovered infrastructure bugs and an
honest, statistically rigorous ledger of ruled-out mechanisms for a still-open execution-time reliability
problem. We believe this combination of a formal criterion, a validated automated tool, and its rigorous
prospective and retrospective application -- reporting negative and incomplete findings with the same care
as positive ones -- is a good fit for T-RO's scope and standards.

**Originality and dual submission.** This manuscript has not been published and is not currently under
review elsewhere.

**AI disclosure.** As noted in the manuscript's Acknowledgment section, the authors used Claude (Anthropic)
to assist with drafting and editing portions of the manuscript text and with statistical analysis
scripting; all experimental design, code implementation, data collection, and scientific claims were
conducted and verified by the authors.

**Suggested reviewers.** [Optional -- add names/affiliations/rationale here if you have specific
suggestions, or delete this paragraph.]

**Conflicts of interest.** [State any individuals who should NOT review this paper due to a conflict of
interest, or state that there are none.]

We thank you and the reviewers for your time and consideration.

Sincerely,

[Corresponding author name -- your supervisor, per your usual convention]
[Department / Faculty]
Universiti Malaya
[Email address]

On behalf of all authors:
[Author list]
