# Hyperparameter Early Stopping

This project aims to predict the final accuracy of a DNN model for given hyperparameters values, without training the DNN model completely.

### Repository organization
* dataset: this folder contains all the dataset files
* preprocess_dataset: this folder contains the preprocessed dataset files generated after running  `preprocessing.ipynb`.
* models: this folder contains the best model generated during model training in `training.ipynb`.
* results: this folder contains the file containing the validation accuracies of the model in the settings asked.

### Running the code
* `preprocessing.ipynb` contains the necessary steps for preprocessing the data. The resulting dataset in stored in the `preprocessed_dataset` folder.
* `training.ipynb`
  * This contains all the necessary code for training and evaluating the model.
  * The best model is saved in `models` folder. This is so the results can be reproduced by loading the model and directly testing the model. To do so, you can skip the training step and run the follwing code to load the model and move on to the testing.

    ```python
    model = LSTM(input_dim, hidden_dim, output_dim)
    state_dict = torch.load(f'{path}/models/final_model.pt')
    model.load_state_dict(state_dict)
    ```
* NOTE:
  * In `training.ipynb`, change the variable `path` to your folder containing all the files. For example,
 
    ```python
    path = '/content/drive/MyDrive/Dataset_hyperEarlyStop'
    ```


  * In `preprocessing.ipynb`, change your working directory to your folder containing all the files. For example, 
  
    ```python
    %cd '/content/drive/MyDrive/Dataset_hyperEarlyStop/'
    ```

    And while, saving the preprocessed dataset, set the path to your folder. For example,

    ```python
    train_loss_df.iloc[train_lines].to_csv("/content/drive/MyDrive/Dataset_hyperEarlyStop/preprocessed_dataset/train_loss_training.csv")
    ```
