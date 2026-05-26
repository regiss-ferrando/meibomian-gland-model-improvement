# Research Paper Rating Methodology

## Overview
This document explains the methodology used to rate research papers in the "Eye AI Research Survey" for relevance to eye-related AI and deep learning research conducted after 2012 (the inception of modern deep learning).

## Rating Scale: 1-10

### Relevance Score Criteria

#### 10/10 - Highly Relevant & Pioneering
**Characteristics:**
- Directly addresses AI/deep learning applications for eye disease detection or eye tracking
- Landmark papers that established methodologies or benchmarks
- High citation count (>500 citations)
- Novel architectures or approaches specifically designed for ophthalmological applications
- Published in top-tier venues (Nature, Science, Lancet, IEEE, top conferences)
- Practical clinical validation and deployment examples
- Addresses critical eye diseases with significant public health impact

**Examples in Dataset:**
- "Automated identification of diabetic retinopathy using deep learning" (Gargeya & Leng, 2017)
- "A deep learning system for detecting diabetic retinopathy across the disease spectrum" (Dai et al., 2021)
- "Glaucoma detection based on deep convolutional neural network" (Chen et al., 2015)
- "Eye tracking for everyone" (Krafka et al., 2016)

---

#### 9/10 - Highly Relevant & Comprehensive
**Characteristics:**
- Core AI/deep learning applications for eye disease or eye tracking
- High citation count (>300 citations)
- Comprehensive review or systematic analysis
- Published in reputable medical or computer science journals
- Strong methodological rigor
- Multi-center studies or large-scale validation
- Addresses emerging research directions

**Examples in Dataset:**
- "Diabetic retinopathy detection through deep learning techniques: A review" (Alyoubi et al., 2020)
- "A critical review on diagnosis of diabetic retinopathy using machine learning and deep learning" (Das et al., 2022)
- "Deep learning for diabetic retinopathy analysis: a review, research challenges, and future directions" (Nadeem et al., 2022)
- "Automatic gaze analysis: A survey of deep learning based approaches" (Ghosh et al., 2023)
- "A review of deep learning techniques for glaucoma detection" (Guergueb & Akhloufi, 2023)
- "Deep learning in glaucoma detection and progression prediction: a systematic review and meta-analysis" (Ling et al., 2025)

---

#### 8/10 - Very Relevant with Strong Contribution
**Characteristics:**
- Direct application of AI/deep learning to eye diseases or eye tracking
- Novel architectural contributions or improvements
- Well-designed studies with good validation
- Published in good-quality journals or conferences
- Citation count 100-300
- Clear methodology and reproducible results
- Contribution to specific subtasks or specialized applications

**Examples in Dataset:**
- "Eye tracking-based diagnosis and early detection of autism spectrum disorder using machine learning and deep learning techniques" (Ahmed et al., 2022)
- "A novel deep learning approach for diagnosing Alzheimer's disease based on eye-tracking data" (Sun et al., 2022)
- "Glaucoma detection and classification using improved U-Net Deep Learning Model" (Kashyap et al., 2022)
- "A hybrid global-local representation CNN model for automatic cataract grading" (Xu et al., 2019)
- "Automatic cataract detection and grading using deep convolutional neural network" (Zhang et al., 2017)
- "Development and validation of a deep learning system to screen vision-threatening conditions in high myopia" (Li et al., 2020)
- "Artificial intelligence applications in ophthalmic optical coherence tomography: a 12-year bibliometric analysis" (Wang et al., 2024)

---

#### 7/10 - Relevant with Moderate Contribution
**Characteristics:**
- Application of AI/deep learning to eye-related problems or eye tracking
- Good methodological approach but possibly limited scope
- Published in peer-reviewed venues
- Citation count 50-100
- Addresses specific eye diseases or specialized applications
- May be geographically or clinically limited
- Contributes to understanding but not necessarily groundbreaking

**Examples in Dataset:**
- "Eye tracking in virtual reality: a broad review of applications and challenges" (Adhanom et al., 2023)
- "Emotion recognition using eye-tracking: taxonomy, review and current challenges" (Lim et al., 2020)
- "Gaze and eye tracking: Techniques and applications in ADAS" (Khan & Lee, 2019)
- "Detection of cataract based on image features using convolutional neural networks" (Weni et al., 2021)
- "Tournament based ranking CNN for the cataract grading" (Jun et al., 2019)
- "Cataract classification based on fundus images using convolutional neural network" (Simanjuntak et al., 2022)
- "Machine learning on cataracts classification using SqueezeNet" (Qian et al., 2018)
- "Classifying three stages of cataract disease using CNN" (Ali et al., 2022)

---

#### 6/10 - Tangentially Relevant
**Characteristics:**
- Related to eye applications but not primarily focused on AI/deep learning
- OR related to AI/deep learning but not specifically for eye applications
- Limited scope or preliminary findings
- May be application-focused without novel technical contribution
- Citation count <50
- Serves as supporting evidence but not core research

**Examples in Dataset:**
- "Using machine learning with eye-tracking data to predict if a recruiter will approve a resume" (Pina et al., 2023)

---

#### 5/10 - Marginally Relevant
**Characteristics:**
- Addresses related topics but with limited direct application
- May combine eye-related research with AI but without deep learning focus
- Early-stage research or proof-of-concept studies
- Low citation count
- Limited scope or methodology

---

#### Below 5/10 - Limited Relevance
**Characteristics:**
- Indirect connection to core topics
- No papers in this survey fall into this category (only highly relevant papers were selected)

---

## Scoring Factors Considered

### Positive Factors Increasing Score:
1. **Deep Learning Focus**: Use of neural networks, CNNs, ResNets, U-Net, etc.
2. **Clinical Relevance**: Direct application to real-world eye diseases
3. **Dataset Scale**: Large, diverse, multi-center datasets
4. **Performance Metrics**: High accuracy, sensitivity, specificity, AUC scores
5. **Validation**: External validation, cross-validation, clinical comparison
6. **Reproducibility**: Open-source code, available datasets
7. **Citation Impact**: Number of citations indicating influence
8. **Publication Venue**: Tier-1 journals and conferences
9. **Novelty**: New architectures, novel applications, innovative approaches
10. **Impact**: Addresses high-burden eye diseases (DR, glaucoma, etc.)

### Negative Factors Decreasing Score:
1. Limited validation (single dataset, no external validation)
2. Small sample sizes
3. Outdated methods (non-deep learning approaches)
4. Limited scope (e.g., single disease, single population)
5. Poor methodological rigor
6. Limited clinical feasibility
7. Reproducibility issues
8. Publication in low-tier venues

---

## Disease/Application Priority Weighting

Papers addressing these conditions received higher priority:

| Priority | Condition/Application | Justification |
|----------|---------------------|----------------|
| 1 (Highest) | Diabetic Retinopathy | Leading cause of vision loss in working-age adults; large-scale screening need |
| 2 | Glaucoma | Silent disease; early detection critical; high disease burden |
| 3 | Cataract | Most common cause of blindness worldwide; high surgical volume |
| 4 | Age-related Macular Degeneration | Leading cause of vision loss in elderly; growing population |
| 5 | Retinopathy of Prematurity | Critical for infant health; requires rapid screening |
| 6 | Myopia/Vision Correction | Rapidly increasing prevalence; prediction/prevention potential |
| 7 | Eye Tracking Technology | Broad applications; accessibility and research value |
| 8 | General Ophthalmology AI | Supporting research, OCT analysis, general imaging |
| 9 | Specialized/Rare Conditions | Limited disease burden; indirect relevance |

---

## Temporal Considerations

- **Post-2012 Focus**: All papers published from 2012 onwards (Deep Learning Era)
- **Recent Papers (2023-2026)**: Given slight boost in relevance due to incorporating latest methodologies
- **Seminal Earlier Papers (2013-2015)**: High score if they established important methodologies or benchmarks
- **Review Papers**: Given high scores if comprehensive, recent, and high-quality

---

## Geographic Diversity

Research was sourced from multiple countries and regions:
- North America (USA, Canada)
- Europe (Germany, Spain, UK)
- Asia (China, India, South Korea, Malaysia, Taiwan, Japan, Singapore)
- Africa (Zambia, Nigeria)
- Middle East & Others (Iran, Iraq)

This geographic diversity ensures representation of different healthcare contexts and populations.

---

## Dataset & Implementation Considerations

Papers included resources for:
- **Public Datasets**: APTOS, Kaggle Eye Disease datasets, ImageNet-based datasets
- **Open-Source Models**: Pre-trained models on HuggingFace, GitHub
- **Reproducibility**: Code availability, detailed methodology descriptions
- **Benchmarking**: Standardized evaluation metrics

---

## Quality Assurance Notes

1. **No Papers Rated Below 6**: Only high-quality, peer-reviewed papers were included
2. **Crosschecking**: Citation counts verified through Google Scholar
3. **Recent Validation**: 2023-2026 papers included latest advancements
4. **Balanced Coverage**: Representation across multiple eye diseases and applications
5. **Rigor Assessment**: Papers evaluated on methodology, validation, and reproducibility

---

## How to Use This Rating System

When evaluating new papers not in this survey:
1. Identify primary focus area (disease/application)
2. Assess level of deep learning involvement
3. Review validation approach and scale
4. Check publication venue and citation count
5. Evaluate clinical feasibility
6. Consider novelty and contribution
7. Assign score based on above criteria

---

## Future Refinements

This methodology may be updated as:
- New high-impact papers are published
- Industry adoption patterns emerge
- Clinical validation standards evolve
- Regulatory guidelines change (FDA/CE approval status)
