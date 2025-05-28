# CSE30401  
Introduction to Data Mining - Final Project

## Project Overview

This project proposes and validates a set of **reference-free evaluation metrics** for summarizing user reviews on e-commerce platforms — where creating gold-standard reference summaries is often impractical or impossible.

User reviews contain rich, real-world insights about products, but their **unstructured format** and overwhelming volume make it difficult for consumers to extract meaningful information efficiently. Traditional summarization systems — especially those designed for structured texts like news articles — often fall short when applied to heterogeneous and informal data like user reviews. Furthermore, most existing evaluation metrics assume the presence of a reference summary, which is **not feasible** for diverse review clusters.

To address this gap, I:

- Cluster user reviews using various **semantic embedding models** and **clustering algorithms**,
- Apply both **extractive** (Centroid, KeyBERT) and **abstractive** (T5) summarization methods to each cluster,
- Propose four new **reference-free evaluation metrics** to assess the quality of review cluster summaries from a user-centric perspective.

### Proposed Evaluation Metrics

| Metric Name                | Description                                                                                     |
|---------------------------|-------------------------------------------------------------------------------------------------|
| `Redundancy`              | Measures sentence-level similarity to penalize repetition                                       |
| `BUYSUMM_FIXED`           | Checks whether key product attributes are mentioned                                             |
| `BUYSUMM_REF`             | Evaluates coverage of important cluster-specific keywords (via KeyBERT)                         |
| `IVD` (Info Value Density)| Assesses how much useful information is packed into each sentence (via aspect-opinion triplets) |

Unlike traditional metrics focused on linguistic accuracy or surface-level similarity, our metrics are designed to evaluate **practical utility**, i.e., whether the summary helps users make better purchase decisions.

I conducted experiments with **60 system configurations** (5 embeddings × 2 clustering methods × 3 summarization models), and confirmed that the proposed metrics capture dimensions of summary quality that are **not reflected in traditional metrics** like ROUGE or BERTScore.

---

## Conclusion & Contributions

This project introduces a novel, **reference-free evaluation framework** specifically designed for **cluster-level summarization of e-commerce reviews** — a domain where traditional metrics fail due to the absence of ground-truth summaries and the unstructured, user-generated nature of the content.

### Key Contributions

- **New Evaluation Metrics**: I propose four quantitative, interpretable metrics — `Redundancy`, `BUYSUMM_FIXED`, `BUYSUMM_REF`, and `IVD` — tailored for assessing the usefulness of review cluster summaries without requiring any reference data.
- **Comprehensive Experimental Validation**: I benchmarked 60 configurations combining various embeddings, clustering, and summarization methods, showing that the proposed metrics successfully capture distinct quality aspects of summaries.
- **User-Centered Evaluation Perspective**: Unlike ROUGE or BLEU, our metrics evaluate how well summaries reflect product attributes, reduce redundancy, and deliver compact, decision-supportive information to users.
- **Metric Independence & Complementarity**: Correlation analysis shows that our metrics (especially `IVD`) measure unique aspects of summary quality, offering a more **holistic evaluation** when combined with traditional metrics.

### Future Directions

- **Human-centered evaluation**: Incorporate user studies to correlate metric scores with perceived summary helpfulness.
- **Product category-specific tuning**: Adapt the evaluation framework to different e-commerce verticals (e.g., electronics, beauty, fashion).
- **Deployment optimization**: Develop lightweight summarization-evaluation pipelines for real-time commercial applications.

By enabling robust and scalable evaluation of review cluster summaries without relying on expensive human annotations, this work lays the foundation for more **intelligent, user-friendly review summarization systems** in real-world e-commerce settings.
