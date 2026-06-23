# Dataset Exploration: Candidates Considered

**Status**: Draft notes from research phase  
**Date**: May 2026

## Datasets Selected for AI Improvement Analysis

### Primary Choice: Meibomian Gland Dataset (MGD-1k) ✓
- **1,000 infrared meibomian gland images**
- **Annotations**: Segmentation masks + graded meiboscore
- **Paper**: Automated quantification of meibomian gland dropout in infrared meibography using deep learning
- **Link**: https://mgd1k.github.io/
- **Status**: **SELECTED** - See `01_MGD1k_Dataset_Selection.md` for details

### Alternative Datasets Evaluated

#### 1. Tear Meniscus Segmentation Dataset (Dry Eye Diagnosis)
- **Link**: https://figshare.com/articles/dataset/Tear_Meniscus_Segmentation_for_Dry_Eye_Diagnosis/
- **Why Considered**: Relevant to dry eye diagnosis
- **Status**: Alternative option (not selected - less specialized than MGD-1k)

#### 2. APTOS 2019 Blindness Detection Dataset (Kaggle)
- **Link**: https://www.kaggle.com/c/aptos2019-blindness-detection
- **Why Considered**: Large public dataset, competition-driven
- **Status**: Alternative option (not selected - too generic, multiple conditions)

### Datasets Removed from Consideration

#### ODIR-5K (Ocular Disease Intelligent Recognition)
- **Reason**: Challenge-style dataset
- **Issues**: 
  - Requires registration for access
  - Designed for competition, not reproducible research
  - Multiple eye diseases in one dataset (less focused)
  - Limited exploration of improvement margin compared to MGD-1k

## Selection Rationale

The **MGD-1k meibomian gland dataset** was chosen because it offers:
1. **Specificity**: Single task with clear clinical focus
2. **Quality**: Expert annotations with high reliability
3. **Manageability**: 1,000 images allows for thorough analysis
4. **Impact**: Addresses a real clinical need (MGD diagnosis)
5. **Research Opportunity**: Published baseline with proven room for improvement

---

**Next**: See `PROJECT_STRATEGY.md` for the detailed implementation strategy
