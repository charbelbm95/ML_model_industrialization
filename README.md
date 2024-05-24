Goal : Industrialize an ML Model

Project Structure Description:
. data folder: datasets folder + data description
. modules: scripts for preproceeing, training and inference
. notebooks: notebooks used for creating the model and then industrializing it 
. joblib_files: files saved during training to be used in inference 
. processed_dfs: dfs saved after initial training and used later after industrializing to make sure that the industrialization process didn't modify the data the model was trained on.

Importance: This is the 1st step after model selection and training towards the application of data science in production , it highlights
the difficulty to industrilize a model but also shows that this process simplifies the deployment and makes it easier to isolate and troubleshoot.
