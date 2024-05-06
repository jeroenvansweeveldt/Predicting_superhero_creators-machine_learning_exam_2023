# Marvel versus DC - predicting superhero creators: machine learning exam
## Objective

For the Machine Learning exam of UAntwerpen's MA of Digital Text Analysis, I was tasked to perform binary classification on the [Superheroes NLP Dataset](https://www.kaggle.com/datasets/jonathanbesomi/superheroes-nlp-dataset/code), predicting whether the creator was DC or Marvel based solely on the _"history_text"_ feature. The objective was to build at least one classical classifier and a neural classifier. Given that my internship and thesis revolved around historical corpus linguistics and involved no machine learning, I did not have the opportunity to familiarise myself properly with RNNs or the Transformers architecture. Therefore, following the exam, I returned to this assignment, to update the preprocessing, the LSTM workflow, and include a DistilBert model to see how it would perform compared to the older technology. 

## Structure
The repository is structured as follows:
* A map **data** containing the original and preprocessed datasets, a dataset called _"gremlins"_, featuring each word that produced gremlins in the original _"Superheroes"_ dataset, and datasets for each classifier used containing test data along with a column that displays whether the prediction was correct.
* A Jupyter notebook containing data exploration and preprocessing.
* A Jupyter notebook containing the preparation, training, and evaluation of the classical models.
* A Jupyter notebook containing the preparation, cross-validation, training, and evaluation of the LSTM model.
* A Jupyter notebook containing the preparation, cross-validation, training, and evaluation of the DistilBert model.

## Disclaimer
To all you fellow DTA students who stumble upon this repository and copied the workflow displayed here for your own exam assignment, know this: you are not supposed to know most of the techniques applied in these notebooks by the time you'll be tasked with the Machine Learning exam, so good luck and have fun explaining those when you'll be asked about them during the exam interview. :3

## The challenge
### Data augmentation

