# Marvel versus DC - predicting superhero creators: machine learning exam
## Objective
For the Machine Learning exam of UAntwerpen's MA of Digital Text Analysis, I was tasked to perform binary classification on the [Superheroes NLP Dataset](https://www.kaggle.com/datasets/jonathanbesomi/superheroes-nlp-dataset/code), predicting whether the creator was DC or Marvel based solely on the _"history_text"_ feature. The objective was to build at least one classical classifier and a neural classifier, using the _Pytorch_ framework. Given that my internship and thesis revolved around historical corpus linguistics and involved no machine learning, I did not have the opportunity to familiarise myself properly with RNNs or the Transformers architecture. Therefore, following the exam, I returned to this assignment, to update the preprocessing, the LSTM workflow, and include a DistilBert model to see how it would perform compared to the older technology. 

## Structure
The repository is structured as follows:
* A map **data** containing the original and preprocessed datasets, a dataset called _"gremlins"_, featuring each word that produced gremlins in the original _"Superheroes"_ dataset, and datasets containing error analysis for each model experimented with.
* A Jupyter notebook containing data exploration and preprocessing.
* A Jupyter notebook containing the preparation, training, and evaluation of the classical models.
* A Jupyter notebook containing the preparation, cross-validation, training, and evaluation of the LSTM model.
* A Jupyter notebook containing the preparation, cross-validation, training, and evaluation of the DistilBert model.
* A Jupyter notebook **utils** containing helper functions for the LSTM model. This code was provided to us by our teacher, and extended by me to make gradient clipping for the model possible.

## Disclaimer
To all you fellow DTA students who stumble upon this repository and copied the workflow displayed here for your own exam assignment, know this: given that you are looking at a version of my exam assignment on steroids, you are not supposed to know yet most of the techniques applied in these notebooks by the time you'll be tasked with the Machine Learning exam, so good luck and have fun explaining those when you're asked about it during the exam interview. :3

## The challenge
With little over 1,000 relevant data points, we have a small dataset. Combined with the high-dimensionality of our feature space with a vocabulary of 54,210 unique words, we are at a high risk of overfitting the data. It is therefore desirable to limit the vocabulary so that the classifiers will be able to learn patterns from the data and generalise better. We can, however, nevertheless expect particular trouble when we are going to use complex models, i.e. deep learning models.

### Data augmentation
#### Filling in NaN values
Increasing the number of data points is a key strategy in combatting overfitting. To start with, I checked the data points with NaN values (not a number, i.e. no data), for both the _"history_text"_ and "_creator"_ columns, and identified using the superhero's real name and/or alias whether they belonged to the DC or Marvel universe. I manually updated these data points with the correct label and/or profile text taken from relevant wikis or fanpages. Furthermore, I made sure that every _"history_text"_ entry contained at least 10 words, expanding the texts in the data points where that was not the case.

#### Chunking and splitting the texts
With the previous preprocessing step complete, the entries in _"history_text"_ have a minimum length of 10 words, while the longest entry contains 14,598 words.

![wordcount_boxplot](https://github.com/jeroenvansweeveldt/Predicting_superhero_creators-machine_learning_exam_2023/assets/98675155/3cba0347-681a-4a0f-bf24-9eca25bc16d6)

Inspired by the methodology used in the research paper [Detecting Incongruity Between News Headline and Body Text via a Deep Hierarchical Encoder](https://arxiv.org/abs/1811.07066), and the [sliding window technique](https://www.geeksforgeeks.org/window-sliding-technique/), I decided to group the texts into chunks of maximum 512 words, as 512 tokens is the maximum length of a sequence BERT Transformers will accept. A condition was programmed to make sure that as soon as a sentence passes the 512 words treshold, the entire sentence would carry over to the next chunk, to avoid chunks ending (and starting) with incomplete sentences.

Afterward, I split these chunks into separate data points. The rationale behind this workflow is this. Since we do not have to predict to which character a _"history_text"_ entry belongs, we do not have to worry about the individual's identity. Moreover, both DC and Marvel work with the concept of a multiverse, meaning that there exist multiple variations of the same character. As a result, the dataset already contained multiple entries for the same character even before preprocessing. In the end, the classifiers will probably base their predictions on words referring to locations or characters unique to Marvel or DC, so the more data points containing these referral words the merrier. 

The operation resulted in an expansion of the dataset from 1,085 data points to 2,317 data points. Better, but still insufficient to comfortably avert the risks of overfitting on the more powerful neural networks.

### Weight balancing
Marvel forms the majority class with 59.13% of labels, versus 40.87% labels assigned to DC.

![class_balance_plot](https://github.com/jeroenvansweeveldt/Predicting_superhero_creators-machine_learning_exam_2023/assets/98675155/77e2f81d-c63d-4b6f-b9ef-fbb74656aa9d)

To counter the data imbalance, I calculated the class weights the models could use for balancing, and thus produce more accurate results.

## Training and results
The machine learning models used for binary classification were _Logistic Regression_, _K Nearest Neighbours_, _Support Vector Machines_, and _Naive Bayes_ for the classical models, and _LSTM_, and _DistilBert_ for the neural models. As _Naive Bayes_ is a [poor estimator](https://scikit-learn.org/stable/modules/naive_bayes.html) due to its nature to take on each feature independent on one another (in our case every word in a profile text), and its error analysis revealed that save for two entries, all of its mistakes were made by attributing profile texts wrongfully to the majority class, I did not include its performance in the table below.

| | LogReg | KNN | SVM | LSTM |  DistilBert |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **f1 macro avg**  | 0.90 | 0.92  | 0.92 | 0.91  | 0.96 |

DistilBert comes out as the clear winner, with a notably higher score than the others, as befits the most technically advanced of all models used in this assignment.
