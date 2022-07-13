## Interpreting Black-box Machine Learning Models for High Dimensional Datasets

<p align="justify">Code and supplementary materials for our paper "Interpreting Black-box Machine Learning Models for High Dimensional Datasets", submitted to IEEE International Conference on Data Science and Advanced Analytics (DSAA'2022). This repo will be updated with more reproducable resources, e.g., models, notebooks, etc.</p>

### Abstract ###
<p align="justify"> Artificial intelligence (AI)-based systems are increasingly deployed in numerous mission-critical situations. Deep neural networks (DNNs) have shown to outperform traditional machine learning (ML) algorithms due to their effectiveness in modelling intricate problems and handling high dimensional datasets in a broad variety of application scenarios. Many reallife datasets, however, are of increasingly high dimensionality, where most features are known to be irrelevant for the task at hand. The inclusion of such features would not only introduce unwanted noise, but also increase computational complexity. Furthermore, due to high non-linearity and dependency among many features, DNN models tend to be complex and perceived as black-box methods because of their not well-understood internal functioning. The internal workings of DNN models are unavoidably opaque as their algorithmic complexity is often simply beyond the capacities of humans to understand the interplay among myriads of hyperparameters. Consequently, a black-box AI system would not allow tracing back how an input instance is mapped to a decision and vice versa. On the other hand, a well-interpretable model can identify statistically significant features and explain the way they affect the model’s outcome. In this paper, we propose an efficient method to improve the interpretability of black-box models for classification tasks in case of high dimensional datasets. We train a complex black-box model on a high dimensional dataset to learn the embeddings on which we classify the instances into specific classes. To decompose the inner working principles of black-box model, we apply probing and perturbing techniques and identify top-k important features. An interpretable surrogate model is then built on top-k feature space to approximate the behaviour of the black-box model. Decision-rules are extracted and explanations are then derived from the surrogate model to explain individual decisions. We test and compare our approach with TabNet, tree-, and SHAP-based interpretability techniques on a number of datasets.</p>

### Datasets
The (Datasets) [https://github.com/AwesomeDeepAI/DeepExplainHidim/tree/main/Datasets] folder contains the following three datasets: 
- **Health advice**: source: Pranay Patil et al., https://github.com/itachi9604/healthcare-chatbot
- **UJIndoorLoc**: source: J. Torres-Sospedra, R. Montoliu, A. Mart´ınez-Uso, T. J. Arnau, M. Benedito-Bordonau, and J. Huerta, "Ujiindoorloc: A new multibuilding and multi-floor database for wlan fingerprint-based indoor localization problems" in 2014 international conference on indoor positioning and indoor navigation (IPIN). IEEE, 2014, pp. 261–270.
- **Forest cover type**: source: J. A. Blackard and D. J. Dean, “Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables,” Computers and electronics in agriculture, vol. 24, no. 3, pp. 131–151, 1999.

The UJIndoorLoc and Forest_cover_type datasets are provided as .zip (due to file size in GitHub). Please unzip them in the respective folders, e.g., 'Datasets/Forest_cover_type/covtype/covtype.csv'. 

As of **Gene expression dataset**, please download the gene expression dataset in pickle format 'TCGA_new_pre_first.pckl' and 'TCGA_new_pre_second.pckl' from https://drive.google.com/drive/u/2/folders/16HR6OoQOeEbfJar3GEZFmk-BKZLeUqCN

### Citation request ###
If you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{DeepExplainHidim,
        title={Interpreting Black-box Machine Learning Models for High Dimensional Datasets},
        author={Anonymous for review},
        conference={submitted to 8th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2022)},
        year={2022}
    }

### Contributing ###
In future, we'll provide an email address, in case readers have any questions.
