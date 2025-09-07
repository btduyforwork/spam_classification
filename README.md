# 📨 Spam Detection with Naive Bayes

This project implements a simple **text classification model** to detect whether a message is **spam** or **ham** using the **Multinomial Naive Bayes** algorithm.  

---

## 📂 Project Overview

### 🔹 Workflow
1. **Load Dataset**  
   - Input: SMS Spam dataset (`ham` / `spam`).  
   - Encode labels (`ham=0`, `spam=1`) using `LabelEncoder`.

2. **Preprocessing**  
   - Lowercasing text  
   - Removing punctuation  
   - Tokenization (split text into words)  
   - Stopword removal (filter out common words like *the, is, of...*)  
   - Stemming/Lemmatization (reduce words to their root forms)

3. **Build Vocabulary**  
   - Create dictionary of unique tokens from training data.

4. **Feature Engineering**  
   - Represent messages as vectors using **Bag-of-Words (BoW)**.  
   - Each row = a message, each column = token count.

5. **Split Dataset**  
   - Training set (70%) → model learns probabilities.  
   - Validation set (20%) → tune hyperparameters (e.g. `alpha`).  
   - Test set (10%) → final evaluation.

6. **Model Training**  
   - Train **Multinomial Naive Bayes** classifier.  
   - Apply Laplace/Lidstone smoothing (`alpha`) to handle zero-probability problem.

7. **Evaluation**  
   - Metrics: **Accuracy** on validation and test sets.  
   - Example:  
     ```
     Val accuracy: 0.98
     Test accuracy: 0.97
     ```

8. **Prediction on New Input**  
   - Convert raw input → preprocessing → BoW vector → predict with trained model.  
   - Ensure feature vector is reshaped to 2D (`reshape(1, -1)`).

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **Google Colab / Jupyter Notebook**  
- **Libraries**:  
  - `numpy`, `pandas` → data handling  
  - `nltk` → text preprocessing (tokenization, stopwords, stemming)  
  - `scikit-learn` → model training & evaluation (`train_test_split`, `MultinomialNB`, `LabelEncoder`, `accuracy_score`)  

---

## 📊 Dataset
- Message Spam Collection dataset.
- Source: https://drive.google.com/uc?id=1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R
- Example:  

| Category | Message                                  |
|----------|------------------------------------------|
| ham      | "Go until jurong point, crazy.. Available only ..." |
| spam     | "Free entry in 2 a wkly comp to win FA Cup final..." |

---

## 🚀 How to Run

1. Clone repo or open notebook in Google Colab.  
2. Run all
