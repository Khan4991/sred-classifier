# Plan for Training an SR&ED Ticket Classifier

## 1. Objective

To build a machine learning model that assists a human reviewer in identifying potentially SR&ED-eligible network change tickets. The primary business goal is to **maximize recall**, ensuring that as few eligible tickets as possible are missed, even at the cost of including some non-eligible tickets for review (lower precision).

## 2. Detailed Plan

The process is divided into five stages:

### Stage 1: Dataset Creation and Labeling

A high-quality dataset is the foundation of the model.

1.  **Load Positive Examples:**
    *   Read `nct_labelled.csv`. These tickets are our confirmed "positive" class.
    *   Assign them a target label of `sred_eligible = 1`.

2.  **Source and Sample Negative Examples:**
    *   Read the master ticket list from `master files/Network Change Tickets (NCT).xlsx`.
    *   Identify all tickets in the master file that are **not** in `nct_labelled.csv`. This large pool will be the source of our "negative" examples.
    *   To handle the class imbalance, randomly sample from this negative pool. Since our goal is high recall, we can tolerate a slightly higher ratio of negatives. A **1:3 ratio** (one positive to three negatives) is a good starting point.

3.  **Combine and Finalize:**
    *   Create a single DataFrame containing all positive examples and the sampled negative examples.
    *   Assign the negative samples a target label of `sred_eligible = 0`.
    *   This combined and labeled DataFrame will be our initial training dataset.

> **Note on Data Contamination:** You've indicated the "true positive" list might be incomplete. This means our negative set is likely contaminated with some unlabeled positive tickets. We will proceed with this assumption, but we will address it in Stage 5 (Iteration and Refinement) using a process called "active learning" to find these hidden positives.

### Stage 2: Feature Engineering

We will convert the ticket data into numerical features for the model, focusing on the fields you identified as potentially valuable.

1.  **Identify Core Features:**
    *   **Text Features:** `Headline/Description`, `Impact Summary`. These are likely to contain the strongest signals.
    *   **Categorical Features:** `Organization`, `Group`, `Day Time/MW`, `ITIL Severity`, `Type of Submission`.

2.  **Preprocess and Vectorize Text Features:**
    *   **Cleaning:** Combine `Headline/Description` and `Impact Summary` into a single text field. Clean this text by converting to lowercase and removing special characters.
    *   **Vectorization:** Use the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique to convert the cleaned text into a numerical matrix. This method gives higher weight to words that are significant for a specific ticket but rare across all other tickets.

3.  **Encode Categorical Features:**
    *   Use **One-Hot Encoding** to convert the categorical features (`Organization`, `Group`, etc.) into binary (0/1) columns that the model can use.

4.  **Combine All Features:**
    *   Merge the TF-IDF text matrix and the one-hot encoded categorical columns into a single, wide feature matrix for training.

### Stage 3: Model Selection

We will start with simple, interpretable models that are highly effective for text classification.

*   **Primary Recommendation:** **Logistic Regression**. It is fast, provides good performance, and its results are relatively easy to interpret.
*   **Alternative to Try:** **Multinomial Naive Bayes**. Another classic, fast, and effective model for text.
*   **Future Options:** If performance is not sufficient, we can explore more complex models like Support Vector Machines (SVM) or Gradient Boosting (XGBoost).

### Stage 4: Training and Evaluation

Our evaluation strategy will be centered around our primary goal of maximizing **Recall**.

1.  **Split the Data:**
    *   Divide the final dataset into a **training set (80%)** and a **testing set (20%)**. The model will learn on the training data and be evaluated on the unseen testing data.

2.  **Train the Model:**
    *   Train the chosen model (e.g., Logistic Regression) on the training set.

3.  **Evaluate for High Recall:**
    *   Use the trained model to make predictions on the testing set.
    *   The **most important metric will be Recall**. This measures the model's ability to find all the *actual* SR&ED tickets. Our goal is to get this as high as possible.
    *   We will also monitor **Precision** and **F1-Score**, but we are willing to accept lower precision to achieve higher recall.
    *   A **Confusion Matrix** will be essential to visualize the trade-off and see the exact number of False Negatives (missed tickets) we are getting.

### Stage 5: Iteration and Refinement

The first model is just a baseline. We will improve it iteratively.

1.  **Error Analysis:**
    *   Manually review the **False Negatives** (the SR&ED tickets the model missed). Look for patterns. Do they use specific language the model is missing? This analysis will provide clues for improving our features.
    *   Also, look at the **False Positives** to understand what is confusing the model.

2.  **Active Learning to Find Hidden Positives:**
    *   To address the data contamination issue, we will use our trained model to make predictions on **all the tickets in the "negative" pool**.
    *   We will sort these "negative" tickets by the model's confidence score. The ones the model thinks are most likely to be positive (despite being labeled negative) will be presented to a human reviewer.
    *   Any tickets the reviewer confirms as eligible will be moved from the negative set to the positive set.
    *   The model will then be **retrained** with this newly enriched data. This loop can be repeated to continuously improve the dataset and the model.

3.  **Hyperparameter Tuning:**
    *   We will adjust the model's settings (e.g., by using a different probability threshold for classification) to find the "sweet spot" that gives us the highest recall while keeping precision at an acceptable level for the reviewer.
