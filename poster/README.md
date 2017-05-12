# Gallery

## Which Features Do I Use? _(Column 1)_

Given labelled data about rooms &hellip;

- Which features are most predictive?
- Empty or occupied?

### Radviz and ParallelCoordinates

Use Yellowbrick Radial Visualizations or Parallel Coordinates to look for class separability!

![occupancy_radviz](figures/occupancy_radviz.png)

![occupancy_parallel_coordinates](figures/occupancy_parallel_coordinates.png)

### Rank2D

Given labelled data about credit card default &hellip;

- Feature relationships?
- Correlations and/or collinearity?

Use Yellowbrick Rank2D for pairwise feature analysis!

![credit_default_covariance_rank2d](figures/credit_default_covariance_rank2d.png)

![credit_default_pearson_rank2d](figures/credit_default_pearson_rank2d.png)



## Working with Text Data _(Column 2)_

Text data is notoriously high-dimensional and hard to visualize. Yellowbrick can help!

### Frequency Distributions

Visualize important word features, before stopwords removal &hellip;

![hobbies_freq_dist](figures/hobbies_freq_dist.png)

&hellip; and after!

![hobbies_freq_dist_stopwords](figures/hobbies_freq_dist_stopwords.png)

### t-SNE

Visualize the distribution of corpus documents in 2 dimensions:
![hobbies_tnse](figures/hobbies_tnse.png)

### Part-of-Speech Tags

How well is our regex part-of-speech tagger labelling with Penn-Treebank tags?

![nursery_nltk_pos_tag](figures/nursery_nltk_pos_tag.png)

![algebra_nltk_pos_tag](figures/algebra_nltk_pos_tag.png)

![recipe_nltk_pos_tag](figures/recipe_nltk_pos_tag.png)

Depends on the text!



## The API _(Column 3)_

### Scikit-Learn

Scikit-Learn has so many models, makes automated model selection very convenient!

```
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn import cross_validation as cv

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(max_depth=5),
    AdaBoostClassifier(),
    GaussianNB(),
]

kfold  = cv.KFold(len(X), n_folds=12)
max([
    cv.cross_val_score(model, X, y, cv=kfold).mean
    for model in classifiers
])
```

Except &hellip;

- search is difficult, high dimensional.
- even with clever optimization, no guaranteed solution.
- time increases exponentially with search space.

The Model Selection Triple (Arun Kumar, et al):

![model_selection_triple](extra_figs/model_selection_triple.png)

### Enter Yellowbrick

Yellowbrick is a new Python library that:

- extends the Scikit-Learn API.
- enhances the model selection process.
- provides visual tools for feature analysis, diagnostics & steering.

### Interface

In Scikit-Learn:

```
# Import the estimator
from sklearn.linear_model import Lasso

# Instantiate the estimator
model = Lasso()

# Fit the data to the estimator
model.fit(X_train, y_train)

# Generate a prediction      
model.predict(X_test)
```

With Yellowbrick:

```
# Import the model and visualizer
from sklearn.linear_model import Lasso
from yellowbrick.regressor import PredictionError

# Instantiate the visualizer
visualizer = PredictionError(Lasso())

# Fit
visualizer.fit(X_train, y_train)

# Score and visualize   
visualizer.score(X_test, y_test)
visualizer.poof()
```

<!-- Result:

![concrete_lassocv_prediction_error](figures/concrete_lassocv_prediction_error.png) -->

### Matplotlib

All Yellowbrick visualizers are built with Matplotlib using the pyplot API. Yellowbrick is not a replacement for other visualization libraries - it's specifically for machine learning.



## Which Model Should I Use? _(Column 4)_

### Prediction Error and Residuals Plot

Visualize the distribution of error to diagnose heteroscedasticity:

![concrete_lassocv_prediction_error](figures/concrete_lassocv_prediction_error.png)

![concrete_ridgecv_residuals](figures/concrete_ridgecv_residuals.png)


### ROCAUC, Classification Report, and Confusion Matrix

ROCAUC helps us see overall accuracy:

![occupancy_random_forest_rocauc](figures/occupancy_random_forest_rocauc.png)

Classification heatmap helps distinguish Type I & Type II error:

![game_nbayes_classification_report](figures/game_nbayes_classification_report.png)

Confusion matrix shows us error on a per-class basis:

![game_maxent_confusion_matrix](figures/game_maxent_confusion_matrix.png)

### Class Balance

What to do with a low-accuracy classifier? Check for imbalance!

![occupancy_random_forest_class_balance](figures/occupancy_random_forest_class_balance.png)

&hellip;that's a visual cue to try stratified sampling, oversampling, or getting more data!



## How Do I Tune My Model? _(Column 5)_

### Elbow Curves and Silhouette Scores

- How do you pick an initial value for k in k-means clustering?
- How do you know whether to increase or decrease k?
- Is partitive clustering the right choice?

Higher silhouette scores mean denser, more separate clusters:

![eight_blobs_kmenas_silhouette](figures/eight_blobs_kmenas_silhouette.png)

The elbow shows the best value of k &hellip; or suggests a different algorithm:

![eight_blobs_kmeans_elbow_curve](figures/eight_blobs_kmeans_elbow_curve.png)

### Alpha Selection

Should I use Lasso, Ridge, or ElasticNet?

![energy_ridgecv_alphas](figures/energy_ridgecv_alphas.png)

Is regularization even working??
