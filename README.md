# Introduction
Code for **IMMENSE**, the multimodal classifier for risky users on social media. Currently, it has been used only with 
Twitter data. <br>

# Usage
You first need to install the dependencies listed in ```requirements.txt```. The file ```parameters.yaml``` contains the
parameters that have to be set according to your dataset and the way you want to run the experiments.

## Files format
The system is multimodal, meaning that it analyzes three different dimensions, respectively the **content**, the **social
relationships** and **spatial relationships**. However, this is not mandatory, and if you don't want to analyze all of 
them you can set to false the fields ```consider_content``` , ```consider_rel``` or ```consider_spat```.

Keep in mind that the dataframe described in the next section always has to be provided, because all the three dimensions
use, in a way or the other, the content. Moreover, it contains the training labels.
### Content
The content must be stored into a ```.csv``` file that has at least the following three columns (the order is not important):
* **id**: The user ID
* **text_cleaned**: This column will contain, for each user, the concatenation of the textual content he/she posted. Therefore, every user is represented by a separate row; 
* **label**: This column tells whether the user associated to it is risky (label=1) or safe (label=0).
The column names used here are indicative, if they are different, just set their names in the [parameters.yaml](parameters.yaml) file, at the fields ```field_id, field_text, field_label```

### Relationships
The graph containing the social or spatial relationships have to be provided in the form of a file containing the edgelist.
The systems expects files containing one row per each edge, and the row will contain the IDs of the users involved in the relation.
In the case of the spatial network, the row will also contain a third value, indicating the closeness among the users. The function
to calculate the closeness among two users starting from their distance is called ```normalize_closeness``` and is
contained in the file  [dataset_class.py](dataset_scripts/dataset_class.py). The values in the rows have to be separated by a separator, 
that you can indicate in the ```parameters.yaml``` file. In the same way you indicate the filenames of the social and spatial networks, 
both for the training and testing. Here's an example of how the social network file should be structured. Suppose you have a tiny social network
with four users: ```user1, user2, user3, user4```. The following edgelist (using "\t" as separator):
```
user1\tuser2
user1\tuser3
user3\tuser1
user2\tuser4
```
Means that ```user1``` follows ```user2``` and ```user3```, that ```user3``` follows ```user1``` and ```user2```
follows ```user4```. Note how the relation is not symmetric. The spatial network, instead should look like this:
```
user1\tuser2\t0.5
user1\tuser3\t0.62
user1\tuser4\t0.3
user2\tuser3\t0
user2\tuser4\0
user3\tuser4\0.1
```
Indicating the spatial closeness among each pair of users. Higher values indicate that the users are closer. This relation is 
symmetric.

## Training and testing
The ```parameters.yaml``` file contains four sections:

* ```dataset_general_params```: for providing general parameters about the train and test dataset
* ```train_dataset_params```: parameters specific to the training dataset
* ```test_dataset_params```: parameters specific to the training dataset
* ```model_params```: parameters for the model to learn

Once the parameter file is completed, you can train the models by running [main_train.py](main_train.py)
and, after the training is completed, test it running [main_test.py](main_test.py). The test will plot a confusion matrix and
print the classification report.


