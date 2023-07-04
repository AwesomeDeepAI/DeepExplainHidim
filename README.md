### Interpreting Black-box Machine Learning Models for High Dimensional Datasets

<p align="justify">Code and supplementary materials for our paper "Interpreting Black-box Machine Learning Models for High Dimensional Datasets", submitted to an international conference. This repo will be updated with more reproducible resources, e.g., models, notebooks, etc.</p>

### Abstract ###
<p align="justify"> Artificial intelligence (AI)-based systems are increasingly deployed in numerous mission-critical situations. Deep neural networks (DNNs) have been shown to outperform traditional machine learning (ML) algorithms due to their effectiveness in modelling intricate problems and handling high-dimensional datasets in a broad variety of application scenarios. Many real-life datasets, however, are of increasingly high dimensionality, where most features are known to be irrelevant to the task at hand. The inclusion of such features would not only introduce unwanted noise but also increase computational complexity. Furthermore, due to high non-linearity and dependency among many features, DNN models tend to be complex and perceived as black-box methods because of their not well-understood internal functioning. The internal workings of DNN models are unavoidably opaque as their algorithmic complexity is often simply beyond the capacities of humans to understand the interplay among myriads of hyperparameters. Consequently, a black-box AI system would not allow tracing back how an input instance is mapped to a decision and vice versa. On the other hand, a well-interpretable model can identify statistically significant features and explain the way they affect the model’s outcome. In this paper, we propose an efficient method to improve the interpretability of black-box models for classification tasks in the case of high-dimensional datasets. We train a complex black-box model on a high-dimensional dataset to learn the embeddings on which we classify the instances into specific classes. To decompose the inner working principles of the black-box model, we apply probing and perturbing techniques and identify top-k important features. An interpretable surrogate model is then built on top-k feature space to approximate the behaviour of the black-box model. Decision rules are extracted and explanations are then derived from the surrogate model to explain individual decisions. We test and compare our approach with TabNet, tree-, and SHAP-based interpretability techniques on a number of datasets.</p>

### Datasets
The [Datasets](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/Datasets) folder contains the following three datasets: 
- **Health advice**: source: Pranay Patil et al., https://github.com/itachi9604/healthcare-chatbot
- **UJIndoorLoc**: source: J. Torres-Sospedra, R. Montoliu, A. Mart´ınez-Uso, T. J. Arnau, M. Benedito-Bordonau, and J. Huerta, "Ujiindoorloc: A new multibuilding and multi-floor database for wlan fingerprint-based indoor localization problems" in 2014 international conference on indoor positioning and indoor navigation (IPIN). IEEE, 2014, pp. 261–270.
- **Forest cover type**: source: J. A. Blackard and D. J. Dean, “Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables,” Computers and electronics in agriculture, vol. 24, no. 3, pp. 131–151, 1999.

As of **Gene expression dataset**, please download the gene expression dataset in pickle format 'TCGA_new_pre_first.pckl' and 'TCGA_new_pre_second.pckl' from https://drive.google.com/drive/u/2/folders/16HR6OoQOeEbfJar3GEZFmk-BKZLeUqCN

### How to use this repo? 
- **Step-1**: the UJIndoorLoc and Forest_cover_type datasets are provided as .zip (due to file size in GitHub). Please unzip them in the respective folders, e.g., 'Datasets/Forest_cover_type/covtype/covtype.csv'. 
- **Step-2**: Then run individual [notebooks](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks):

    - [AutoML_ForestCoverType.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/AutoML_ForestCoverType.ipynb): AutoML (i.e., Pycaret)-based classification on forest cover type dataset. 
    - [AutoML_GE.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/AutoML_GE.ipynb): AutoML (i.e., Pycaret)-based classification on gene expression dataset. 
    - [AutoML_Symptom_Precaution.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/AutoML_Symptom_Precaution.ipynb): AutoML (i.e., Pycaret)-based classification on symptom precaution dataset. 
    - [AutoML_UJIndoorLoc.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/AutoML_UJIndoorLoc.ipynb): AutoML (i.e., Pycaret)-based classification on UJIndoorLoc dataset. 
    - [SAN_CAE_Gene_Expression.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/SAN_CAE_Gene_Expression.ipynb): convolutional autoencoder (CAE)-based feature embedding and classification using self-attention network (SAN) on gene expression dataset. An example for another dataset will be uploaded soon. 
    - [TabNet_Forest_Cover_Type.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/TabNet_Forest_Cover_Type.ipynb): classification using TabNet on forest cover type dataset. 
    - [TabNet_GE.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/TabNet_GE.ipynb): classification using TabNet on gene expression dataset. 
    - [TabNet_Symptom_Precaution.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/TabNet_Symptom_Precaution.ipynb): classification using TabNet on symptom precaution dataset.
    - [TabNet_UJIndoorLoc.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/TabNet_UJIndoorLoc.ipynb): classification using TabNet on  UJIndoorLoc dataset.
    - [XGBoost_Forest_Cover_Type.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/XGBoost_Forest_Cover_Type.ipynb): classification using XGBoost on forest cover type dataset. 
    - [XGBoost_GE.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/XGBoost_GE.ipynb): classification using XGBoost on symptom precaution dataset.
    - [XGBoost_Symptom_Precaution.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/XGBoost_Symptom_Precaution.ipynb): classification using XGBoost on symptom precaution dataset.
    - [XGBoost_UJIndoorLoc.ipynb](https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/notebooks/XGBoost_UJIndoorLoc.ipynb): classification using XGBoost on UJIndoorLoc dataset. 

### Citation request ###
If you use the code of this repository in your research, please consider citing the following papers:

    @inproceedings{DeepExplainHidim,
        title={Interpreting Black-box Machine Learning Models for High Dimensional Datasets},
        author={Anonymous for review},
        conference={under review in an international conference},
        year={2022}
    }

### Contributing ###
In future, we'll provide an email address, in case readers have any questions.
