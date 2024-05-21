```markdown
# Fraud Detection in Supply Chains Using Kolmogorov Arnold Networks

This project, titled "Enhancing Fraud Detection in Supply Chains with Kolmogorov Arnold Networks: A Comparative Analysis with Multi-Layer Perceptrons (Initial Findings and Methodology)," aims to enhance fraud detection in supply chains by leveraging Kolmogorov Arnold Networks (KANs) and contrasting their performance with traditional Multi-Layer Perceptrons (MLPs).

## Project Overview

The study explores the application of Kolmogorov Arnold Networks (KANs) in detecting fraud within supply chains, emphasizing their interpretability and adaptability compared to traditional Multi-Layer Perceptrons (MLPs). KANs utilize learnable activation functions, enhancing model transparency and enabling more effective fraud detection.

## Repository Structure

- **Article**: Contains the detailed study and findings.
- **Classification**: Scripts and models related to the classification tasks.
- **PCA_and_other_approaches**: Alternative approaches and dimensionality reduction techniques.
- **Regression**: Scripts and models related to regression tasks.
- **DataCoSupplyChainDataset.zip**: The dataset used for the study.
- **DescriptionDataCoSupplyChain.csv**: Description of the dataset features.
- **README.md**: This file.

## Data Collection

The dataset comprises transactional data from a supply chain, including features such as transaction amounts, product categories, customer information, and timestamps. The dataset can be found [here](https://data.mendeley.com/datasets/8gx2fvg2k6/5).

## Data Processing

### Data Cleaning
- Removed unnecessary columns and rows with missing values in critical fields.
- Combined customer names into a single column.
- Split order dates into separate year, month, day, and hour columns.

### Feature Engineering
- Created a target variable named 'fraud' based on the 'Order Status' category.
- Unified first and last names into a 'Cust_Full_Name' column.
- Formatted date columns for consistency and time-based analysis.

### Encoding Categorical Variables
- Identified and label-encoded categorical columns to convert them into numerical values.

### Handling Class Imbalance
- Applied the Adaptive Synthetic (ADASYN) algorithm to balance the training data.

### Converting Data to Tensors
- Split the dataset into training and test sets.
- Converted the preprocessed data into PyTorch tensors.

## Model Development

### KAN Setup & Layer Configuration
- Constructed using the KAN library with learnable activation functions.
- Configured with an input layer, six hidden neurons, and output neurons for classification tasks.

### Training Process
- Utilized the LBFGS optimizer for high-dimensional problems.
- Trained for 11 epochs, tracking both training and test accuracy.
- Saved visual representations of the training process.

## Evaluation Metrics

- Achieved a train accuracy of 60% and a test accuracy of 98%.
- Derived a symbolic formula for the decision boundary with train and test accuracies of 59.3% and 97% respectively.
- Visualized model activations and pruned less significant connections for efficiency.

## Analysis and Insights

- **Accuracy**: Substantial improvement in test accuracy.
- **Formula Derivation**: Captured underlying patterns of fraudulent transactions.
- **Visualization and Pruning**: Ensured efficient model performance while maintaining accuracy.

## Suggestions

- **Compute Resources**: Larger datasets require greater compute power, suggesting the need for GPU usage.
- **Ideal Applications**: Suitable for various fields including pharmaceutical, manufacturing, market research, energy sector, environmental science, and legal compliance.

## Requirements

The project requires the following dependencies:

Requirements

```
# python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
```


For installation of these packages, please refer to the [installation guide](https://github.com/KindXiaoming/pykan?tab=readme-ov-file#installation).

## Contributing

Contributions to the project are welcome. Please fork the repository, make your changes, and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

## References

- Soori, M., & Arezoo, B. (2023). Artificial Neural Networks (ANNs) in supply chain management: Opportunities and challenges. Journal of Economy and Technology, 18(3), 87-102. DOI
- Das, S. (2023). Artificial Neural Networks for Fraud Detection in Supply Chain Analytics: MLPClassifier and Keras. GitHub repository
- Liu, Z., et al. (2024). Kolmogorov-Arnold Networks. arXiv preprint

For more details, refer to the [Article](https://chrisd-7.github.io/ChrisDSilva/assets/files/articles/KAN-Article/KANArticle.html) directory.

## Citation

```python
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

## To Cite me

```python
@project{ChrisD-7,
  title={Enhancing Fraud Detection in Supply Chains with Kolmogorov Arnold Networks: A Comparative Analysis with Multi-Layer Perceptrons (Initial Findings and Methodology)},
  author={Chris DSilva},
  year={2024},
  howpublished={\url{https://github.com/ChrisD-7/Fraud-Detection-in-Supply-Chains-with-Kolmogorov-Arnold-Networks/blob/main/README.md}}
}
```
```
