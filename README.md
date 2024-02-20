# time-to-event

## Papers to Read

- [x] DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network, BMC Medical Research Methodology, 2018 ([see link](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1))
- [ ] DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks, AAAI, 2018 ([see link](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit))
- [ ] X-CAL: Explicit Calibration for Survival Analysis, NeurIPS, 2020 ([see link](https://papers.nips.cc/paper/2020/file/d4a93297083a23cc099f7bd6a8621131-Paper.pdf))
- [ ] An Effective Meaningful Way to Evaluate Survival Models, ICML, 2023 ([see link](https://proceedings.mlr.press/v202/qi23b/qi23b.pdf))
- [ ] A General Framework for Visualizing Embedding Spaces of Neural Survival Analysis Models Based on Angular Information, CHIL, 2023 ([see link](https://proceedings.mlr.press/v209/chen23b/chen23b.pdf))

## Baseline Results

| Loss function |      Test CI      |
|:--------------|:-----------------:|
| MSEUncensored | 0.9259 +/- 0.0066 |
| MSEHinge      | 0.9564 +/- 0.0010 |
| DeepSurv      | 0.9565 +/- 0.0006 |
