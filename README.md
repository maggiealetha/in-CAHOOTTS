# in-CAHOOTTS: A Neural ODE Framework for Dynamic Gene Regulatory Network Inference and Temporal Modeling

**in-CAHOOTTS** ( gene regulatory network **in**ference for **C**ontext **A**ware **H**ybrid neural-**O**DEs **o**n **T**ranscriptional **T**ime-series **S**ystems) is a biophysically-motivated neural ordinary differential equation framework that simultaneously infers gene regulatory networks and predicts gene expression dynamics from single-cell time-series data.

## Key Features

- **Biophysical Decomposition**: Separates gene expression changes into mRNA transcription and degradation components
- **Interpretable Architecture**: Maintains full biological interpretability while achieving high predictive accuracy
- **Long-term Prediction**: Enables unprecedented temporal extrapolation (30x beyond training data)
- **Transcription Factor Activity Inference**: Estimates latent TF activities that drive regulatory responses
- **Prior Knowledge Integration**: Incorporates known regulatory interactions to guide network inference

<img width="1920" height="1080" alt="figure_1_png" src="https://github.com/user-attachments/assets/c876e328-2c5e-4679-9d4a-2845249322e5" />

## Data

The single-cell RNA sequencing data used in this study is available from GEO under accession number GSE242556. The dataset contains 173,361 *Saccharomyces cerevisiae* cells sampled over 80 minutes during rapamycin treatment.

For detailed data preprocessing steps, see the Methods section of our [bioRxiv preprint](http://bit.ly/4pA2MvN).
