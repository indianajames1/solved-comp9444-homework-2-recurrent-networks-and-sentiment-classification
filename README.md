Download Link: https://assignmentchef.com/product/solved-comp9444-homework-2-recurrent-networks-and-sentiment-classification
<br>
<h1>Part 1</h1>

For Part 1 of the assignment, you should work through the file part1.py and complete the functions where specified.

<h1>Part 2</h1>

For Part 2, you will develop several models to solve a text classification task on movie review data. The goal is to train a classifier that can correctly identify whether a review is positive or negative. The labeled data is located in data/imdb/aclimdb and is split into train (training) and dev (development) sets, which contain 25000 and 6248 samples respectively. For each set, the balance between positive and negative reviews is equal, so you don’t need to worry about class imbalances.

You should take at least 10 minutes to manually inspect the data so as to understand what is being classified. In the entire collection, no more than 30 reviews are allowed for any given movie because reviews for the same movie tend to have correlated ratings. Further, the train and dev sets contain a disjoint set of movies, so no significant performance is obtained by memorizing movie-unique terms and their association with observed labels. In the labeled train/dev sets, a negative review has a score &lt;= 4 out of 10, and a positive review has a score &gt;= 7 out of 10. Thus reviews with more neutral ratings are not included.

The provided file part2.py is what you need to complete. This code makes heavy use of torchtext, which aims to be the NLP equivelent to torchvision. It is advisable to develop a basic understanding of the package by skimming the documentation <a href="https://torchtext.readthedocs.io/en/latest/">here</a><a href="https://torchtext.readthedocs.io/en/latest/">,</a> or reading the very good tutorial <a href="http://anie.me/On-Torchtext/">here</a>.

Since this is not an NLP course, the following have already been implemented for you:

Dataloading: a dataloader has been provided in imdb_dataloader.py. This will load the files into memory correctly.

Preprocessing: review strings are converted to lower case, lengths of the reviews are calculated and added to the dataset. This allows for dynamic padding.

Tokenization: the review strings are broken into a list of their constituent words. Vectorization: words are converted to vectors. Here we use 50-dimensional GloVe embeddings.

Batching: We use the BucketIterator() provided by torchtext so as to create batches of similar lengths. This isn’t necessary for accuracy but will speed up training since the total sequence length can be reduced for some batches.

Glove vectors are stored in the .vector_cache directory.

You should seek to understand the code provided as it will be a good starting point for part 3. Additionally, the code is structured to be backend-agnostic. That is, if a GPU is present, it will automatically be used, if one is not, the CPU will be used. This is the purpose of the .to(device) function being called on several operations.

For all tasks in this part, if arguments are not specified assume PyTorch defaults.

<h1>Task 1: LSTM Network</h1>

Implement an LSTM Network according to the function docstring. When combined with an appropriate loss function this model should achieve ~81% when run using the provided code.

<h1>Task 2: CNN Network</h1>

Implement a CNN Network according to the function docstring. When combined with an appropriate loss function this model should achieve ~82% when run using the provided code. Task 3: Loss function

Define a loss function according to the function docstring.

<h1>Task 4: Measures</h1>

Return (in the following order), the number of true positive classifications, true negatives, false positives and false negatives. True positives are positive reviews correctly identified as positive. True negatives are negative reviews correctly identified as negative. False positives are negative reviews incorrectly identified as positive. False negatives are postitive reviews incorrectly identified as negative.

<h1>Part 3</h1>

The goal of this section is to simply achieve the highest accuracy you can on a holdout test set (i.e. a section of the dataset that we do not make available to you, but will test your model against).

You may use any form of model and preprocessing you like to achieve this, provided you adhere to the constraints listed below.

The provided code part3.py is essentially the same as part2.py except that it reports the overall accuracy, and at the end of training it saves the model in a file called model.pth (which you will need to submit). A good starting point would be to copy the relevant sections of code from your best model for part2.py into part3.py.

Your code must be capable of handling various batch sizes. You can check this is working ok with the submission tests. The code provided in part3.py already does this.

You can modify and change the code however you would like, however you MUST ensure that we can load your code to test it. This is done in the following way:

<ol>

 <li>Import and create and instance of your network from the py file you submit.</li>

 <li>Restore this network to its trained state using the state-dict you provide.</li>

 <li>Load a test dataset, preprocessing each sample using the text_field you specify in your</li>

</ol>

PreProcessing class.

<ol start="4">

 <li>Feed this dataset into your model and record the accuracy.</li>

</ol>

You should check the docs on the torchtext.data.Field class to understand what you can and canâ€&#x2122;t do to the input.

Specific to preprocessing, you may add a post-processing function to the field, as long as that function is also declared in the Preprocessing class. You may also add a custom tokenizer, stopwords, etc. Note that none of this is necessarily required, but it is possible.

You may wish to carry out some data augmentation. This is because in practice more data will outperform a better model. Data augmentation (transforming the data you have been provided and creating a new sample with the same label) is allowed. You are allowed to modify the main() function to create additional data in place. You may not call any remote API’s when doing this. Assume the test environment has no internet connection.

You may NOT download or load data other than what we have provided. If we find your submitted model has been trained on external data you will receive a mark of 0 for the assignment.

Marks for part 3 will be based primarily on the accuracy your model achieves on the unseen test set.

When you submit part 3, in addition to the standard checks that we can run and evaluate your model, you will also see an accuracy value. This is the result of running your model on a very small number of held-out training examples (~600). These samples can be considered representative of the final test set, however the final accuracy will be calculated from significantly more samples (~18 000). The submission test should take no longer than 10s to run.