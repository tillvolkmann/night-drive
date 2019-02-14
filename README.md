## Night Drive – Augmenting data sets for autonomous driving with Generative Adversarial Networks ##

### The Motivation ###
Visual perception models within driver-assistance systems or autonomous driving applications need a large amount of labeled image data. For a detection task for instance, we consider bounding boxes around relevant objects as the labels whereas for a classification task the label could be a class name or a property assigned to an image.

Usually, the quality in performing these tasks should stay the same, regardless of the lighting situation, e. g. day-time vs. night-time, the current weather conditions or the region the data has been recorded. To achieve this, a properly balanced data set across all possible situations and conditions is crucial. However, since data acquisition and labeling is costly regarding time and money, many real world projects end up using somewhat unbalanced data sets that then are used to train slightly biased models.

### The Project ####
As our portfolio project for DSR Batch #17, we want to analyze the benefits of a possible solution to the problem described above by choosing one particular use-case. Assuming, that we have a data set containing less night-time images than day-time images, we want to train a Generative Adversarial Network to generate more night-time images from existing day-time images. This task is also known as domain transfer.
If we succeed, we would get the labels within the night-time images for free, since they originated from daytime-images where the labels are known and the spatial construction of the scene within the generated images should stay the same.

To quantify the benefits of the generated images, they will be augmented to the training data set of two models: a pedestrian detector and weather classifier. Both models will be trained without and with the augmented training data set. Since validation and test will be done on unchanged data sets containing only daytime and non-generated night-time images, we should be able to give a good estimate, if and how much our approach could balance the performance of the models.

### The Team ###
The project will be tackled in a team of two. We are @csoehnel and @tillvolkmann.

