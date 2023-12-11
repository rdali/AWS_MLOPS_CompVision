# AWSMLOpsAssignment
Code repo for AWS MLOPS SageMaker Pilot


## Issues and Caveats for project:

This project was built based on a small Kaggle dataset to test out SageMaker.

The dataset is really small and there can overlap between the [training dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
) and the [prediction dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri). This should not happen in a production project.

This project was run based on notebooks and useful functions were copied from notebook to another. This should also NOT happen in a production project. Code should be centralized and reused.

This project dealt with a binary classification of tumor vs Normal brain. This is an extrememly simple modeling that was mainly to test ipelining on AWS. There are many brain diseases that are not brain cancer that would also need to be included for a more holistic dataset.

This project detected brain cancers in 2D images. In reality MRIs and brain cancers as well as brain normal structured are 3D. training a model on the full volume of the brain would be more useful.