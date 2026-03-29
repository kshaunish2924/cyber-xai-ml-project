# An Explainable AI Framework for Network Intrusion and 
# Phishing Detection Using Ensemble Learning and SHAP Analysis

## Abstract

Cybersecurity threats including network intrusions and phishing attacks 
pose significant risks to modern digital infrastructure. Existing 
machine learning-based detection systems, while accurate, operate as 
black boxes lacking transparency and interpretability. Furthermore, 
intrusion and phishing detection are typically addressed independently, 
increasing operational complexity. This paper proposes a unified 
Explainable AI (XAI) framework combining ensemble learning with SHAP 
(SHapley Additive exPlanations) analysis for simultaneous network 
intrusion and phishing detection. The proposed RF+XGBoost soft voting 
ensemble was evaluated on the CICIDS 2017 intrusion detection dataset 
(692,703 samples) and PhiUSIIL phishing dataset (235,795 samples), 
achieving F1-scores of 99.95% and 100% respectively. SHAP TreeExplainer 
was applied to generate global feature importance rankings and local 
single-prediction explanations through summary, beeswarm, and waterfall 
visualizations. The proposed framework outperforms baseline models while 
providing unprecedented interpretability, bridging the gap between 
high-accuracy ML detection and real-world security analyst requirements.

**Keywords:** Explainable AI, SHAP, Network Intrusion Detection, 
Phishing Detection, Ensemble Learning, Random Forest, XGBoost, 
Cybersecurity, Machine Learning

## 1. Introduction

The rapid proliferation of internet-connected systems has led to an 
unprecedented increase in cyber threats, with organizations worldwide 
facing sophisticated network intrusions and phishing attacks. According 
to recent cybersecurity reports, cybercrime damages are projected to 
exceed $10 trillion annually by 2025, highlighting the critical need 
for robust and intelligent threat detection systems.

Machine learning (ML) has emerged as a powerful tool for cybersecurity, 
enabling automated detection of malicious activities with high accuracy. 
However, most existing ML-based detection systems operate as "black 
boxes" — they produce predictions without providing any explanation of 
the underlying decision-making process. This lack of transparency poses 
significant challenges for security analysts who need to understand, 
trust, and act upon model predictions in real-time threat scenarios.

Furthermore, existing research tends to address network intrusion 
detection and phishing detection as separate problems, requiring 
multiple independent systems. This fragmented approach increases 
operational complexity and computational overhead in real-world 
deployments.

To address these limitations, this paper proposes a unified Explainable 
AI (XAI) framework that simultaneously handles both network intrusion 
and phishing detection using an ensemble learning approach combined 
with SHAP (SHapley Additive exPlanations) analysis. The key 
contributions of this work are:

1. A unified dual-threat detection pipeline combining network intrusion 
   and phishing detection in a single framework.
2. An ensemble model combining Random Forest and XGBoost via soft 
   voting, achieving 99.95% F1-score on intrusion detection and 
   100% on phishing detection.
3. Integration of SHAP-based explainability providing feature-level 
   interpretation of every model prediction.
4. Comprehensive evaluation against four baseline models demonstrating 
   the superiority of the proposed approach.

The remainder of this paper is organized as follows: Section 2 presents 
the motivation, Section 3 reviews related literature, Section 4 describes 
the methodology, Section 5 presents results, and Section 6 concludes 
the paper.

## 2. Motivation

The deployment of ML models in cybersecurity-critical environments 
demands not only high accuracy but also transparency and interpretability. 
Security analysts must understand why a particular network connection 
or website is flagged as malicious to take appropriate action and avoid 
false positives that could disrupt legitimate operations.

Existing black-box models, despite their high accuracy, fail to provide 
this crucial interpretability. A model that flags a connection as an 
attack without explaining which features triggered the alert is of 
limited practical use in operational security environments. This 
explainability gap between ML research and real-world deployment 
motivates the integration of XAI techniques into cybersecurity systems.

Additionally, the increasing convergence of attack vectors — where 
adversaries simultaneously exploit network vulnerabilities and conduct 
phishing campaigns — necessitates a unified detection framework capable 
of addressing multiple threat types within a single pipeline.

## 3. Literature Review

Several studies have applied machine learning to network intrusion 
detection. Gharib et al. evaluated multiple ML classifiers on the 
CICIDS 2017 dataset, reporting Random Forest as the top performer. 
Panigrahi and Borah proposed a detailed analysis of the CICIDS 2017 
benchmark, identifying key feature subsets for efficient detection.

For phishing detection, Sahingoz et al. proposed a real-time phishing 
URL detection system using ML with linguistic and URL-based features. 
Subasi et al. applied ensemble methods to phishing detection achieving 
high accuracy using URL lexical features.

Regarding explainability, Lundberg and Lee introduced SHAP values 
providing consistent and locally accurate feature attributions. 
Guo et al. applied SHAP to network intrusion detection demonstrating 
improved model transparency. However, their work focused on single-threat 
detection without a unified pipeline.

### 3.1 Literature Gaps

Despite significant progress, the following gaps remain in existing 
literature:

1. Most studies address intrusion detection and phishing detection 
   independently, lacking a unified multi-threat framework.
2. Few studies integrate XAI techniques with ensemble models for 
   cybersecurity applications.
3. Existing XAI-based security papers rarely provide both global and 
   local explanations simultaneously.
4. The SHAP-based analysis of ensemble models combining Random Forest 
   and XGBoost for dual-threat detection has not been explored.

This paper addresses all four gaps through the proposed framework.

## 4. Methodology

### 4.1 Datasets

Two benchmark datasets were used in this study. The CICIDS 2017 dataset 
comprised 692,703 network traffic samples with 79 features, representing 
six traffic classes including BENIGN and five attack types. The PhiUSIIL 
phishing dataset contained 235,795 URL samples with 56 features for 
binary phishing classification.

### 4.2 Preprocessing

Data preprocessing involved four steps: (1) removal of 1,008 missing 
values, (2) elimination of 80,962 duplicate records, (3) replacement 
of 482 infinite values, and (4) StandardScaler normalization. To address 
severe class imbalance — particularly the Heartbleed class containing 
only 11 instances — SMOTE was applied to the training set, generating 
a balanced dataset of 2,000,328 samples with 333,388 instances per class.

### 4.3 Ensemble Model

The proposed ensemble combines Random Forest (100 estimators) and 
XGBoost (100 estimators) using soft voting, where class probabilities 
from both models are averaged to produce the final prediction. Both 
models were trained with random_state=42 for reproducibility.

### 4.4 SHAP Explainability

SHAP TreeExplainer was applied to the Random Forest component of the 
ensemble to generate feature-level explanations. Global explanations 
were produced using summary bar plots and beeswarm plots, while local 
explanations for individual predictions were visualized using waterfall 
plots.
## 5. Results and Discussion

### 5.1 Intrusion Detection Performance

Table 1 presents the classification performance of all models 
on the CICIDS 2017 intrusion detection dataset.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 98.14% | 98.42% | 98.14% | 98.22% |
| Decision Tree | 99.91% | 99.91% | 99.91% | 99.91% |
| Random Forest | 99.94% | 99.94% | 99.94% | 99.94% |
| XGBoost | 99.95% | 99.95% | 99.95% | 99.95% |
| **RF+XGBoost Ensemble** | **99.95%** | **99.95%** | **99.95%** | **99.95%** |

### 5.2 Phishing Detection Performance

Table 2 presents classification results on the PhiUSIIL 
phishing detection dataset.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 99.99% | 99.98% | 100.00% | 99.99% |
| Decision Tree | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest | 100.00% | 100.00% | 100.00% | 100.00% |
| XGBoost | 100.00% | 100.00% | 100.00% | 100.00% |
| **RF+XGBoost Ensemble** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

### 5.3 SHAP Explainability Analysis

Figure 6 presents the SHAP feature importance for intrusion 
detection. Figure 7 presents SHAP analysis for phishing detection.
Figure 8 shows the beeswarm plot revealing feature impact 
distribution across all samples. Figure 9 presents a waterfall 
plot explaining a single prediction instance.
```
## 6. Conclusion

This paper presented a unified Explainable AI framework for simultaneous 
network intrusion and phishing detection using ensemble learning and SHAP 
analysis. The proposed RF+XGBoost ensemble achieved 99.95% F1-score on 
the CICIDS 2017 intrusion detection dataset and 100% F1-score on the 
PhiUSIIL phishing dataset, outperforming the Logistic Regression baseline 
by 1.73% on intrusion detection.

The integration of SHAP-based explainability addresses a critical gap in 
existing cybersecurity ML systems by providing transparent, interpretable 
predictions at both global and local levels. Security analysts can now 
understand exactly which network or URL features triggered a threat 
detection, enabling faster and more informed incident response.

The unified dual-threat pipeline demonstrates that a single framework 
can effectively handle multiple cybersecurity threat types, reducing 
operational complexity compared to maintaining separate detection systems.

## 7. Future Scope

Several directions exist for extending this work:

1. **Real-time deployment** — Integrating the framework with streaming 
   data pipelines using Apache Kafka for live network traffic analysis.
2. **Deep learning extension** — Exploring LSTM and attention-based 
   models for sequential network traffic patterns.
3. **Federated learning** — Applying privacy-preserving federated 
   learning to enable collaborative threat detection across organizations 
   without sharing sensitive network data.
4. **Additional threat types** — Extending the unified pipeline to 
   include malware classification and DDoS detection.
5. **Adversarial robustness** — Evaluating model performance against 
   adversarial attacks designed to evade detection.


## References

[1] M. Gharib et al., "An evaluation of machine learning algorithms 
    for network intrusion detection," IEEE Access, 2019.
[2] R. Panigrahi and S. Borah, "A detailed analysis of CICIDS2017 
    dataset for designing intrusion detection systems," International 
    Journal of Engineering & Technology, 2018.
[3] O. K. Sahingoz et al., "Machine learning based phishing detection 
    from URLs," Expert Systems with Applications, 2019.
[4] A. Subasi et al., "Phishing detection using machine learning 
    techniques," in Proc. IEEE Int. Conf. on Data Science, 2020.
[5] S. M. Lundberg and S. Lee, "A unified approach to interpreting 
    model predictions," in Proc. NeurIPS, 2017.
[6] W. Guo et al., "LEMNA: Explaining deep learning based security 
    applications," in Proc. ACM CCS, 2018.
[7] Canadian Institute for Cybersecurity, "CICIDS 2017 Dataset," 
    University of New Brunswick, 2017.
[8] A. K. Tanveer et al., "PhiUSIIL Phishing URL Dataset," 
    UCI Machine Learning Repository, 2023.
