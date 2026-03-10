# Hybrid Recommendation System for Beauty E-Commerce

![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)

This repository contains the complete codebase for a hybrid recommendation system designed for a Turkish beauty e-commerce platform. The project leverages a combination of collaborative filtering (SVD), content-based filtering (TF-IDF), and strategic business rules to deliver a nuanced, 5-slot recommendation experience. It moves beyond generic "people also bought" lists to address specific user contexts like replenishment, cross-selling, and new product discovery.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [The Solution: A 5-Slot Hybrid Architecture](#the-solution-a-5-slot-hybrid-architecture)
- [Key EDA Findings](#key-eda-findings)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Technologies Used](#technologies-used)
- [Team](#team)
- [License](#license)

---

## Problem Statement

Standard recommendation engines often fail in the beauty e-commerce space for several key reasons:

1.  **Two Distinct Economies**: The data reveals two separate demand curves. A small set of high-priced, infrequently purchased luxury items (like *Parfüm*) drives revenue, while a large volume of low-priced, frequently purchased essentials (like *Makyaj* and *Kişisel Bakım*) drives transactions. A generic engine cannot effectively serve both of these user behaviors.
2.  **Extreme Data Sparsity**: Over 50% of products in the catalog have been purchased fewer than 5 times, and 18.8% have been purchased only once. This "long tail" makes it impossible for pure collaborative filtering models like SVD to generate recommendations for a significant portion of the inventory.
3.  **Lack of Context**: Generic models are blind to user context. They don't know if a user needs to replenish a routine item, is looking for a complementary product to go with something they just bought, or is open to discovering a new brand.

## The Solution: A 5-Slot Hybrid Architecture

To address this, we designed a hybrid, multi-slot recommendation engine where each slot has a specific strategic purpose. This transforms the recommender from a simple list into a structured, context-aware module.

| Slot | Type | Logic |
| :--- | :--- | :--- |
| **1** | **Smart Replenishment** | Finds a frequently purchased product that is past its average consumption period. |
| **2** | **Complementary Product A** | Recommends a product-line-similar item from a complementary category (e.g., the matching conditioner for a shampoo). |
| **3** | **Complementary Product B** | Same as Slot 2, but from a different complementary category to ensure variety. |
| **4** | **New Product Discovery** | Best SVD-scored new product from a category not yet represented in the list. |
| **5** | **Brand Exploration** | Best SVD-scored product from a brand the customer has never purchased, within their price tier. |

This architecture is governed by global filters for **Price Tier Consistency** (users are only shown products in their inferred price range) and runs only for registered members.

## Key EDA Findings

Our model design is directly informed by four key findings from the Exploratory Data Analysis:

1.  **Two Demand Curves Exist**: Only 7 of the top 20 products appeared in both the revenue and transaction rankings. This confirmed that luxury and essential products have separate buying logics, requiring different recommendation strategies.
2.  **Sparsity Requires a Hybrid Model**: With 50.1% of products having fewer than 5 purchases, we validated that a pure collaborative filtering approach would fail. This necessitated the use of content-based (TF-IDF) and rule-based logic as fallbacks.
3.  **Brand Loyalty is Nuanced**: Only 7 of the top 15 brands appeared in both top revenue and volume lists. Users stick to price tiers, validating the need for the Price Tier Consistency filter.
4.  **Co-purchase Signals are Strong**: We found statistically significant cross-category co-purchase patterns (e.g., Parfüm buyers also buying Makyaj), which provides the empirical foundation for the complementary recommendation slots.

## Repository Structure

```
.
├── Data Cleaning & Feature Engineering.ipynb
├── EDA.ipynb
├── Apriori Algorithm.ipynb
├── hybrid_beauty_product_recommender.ipynb
├── prediction_model.ipynb
├── complementary_map.json
└── README.md
```

-   **`Data Cleaning & Feature Engineering.ipynb`**: Merges the three raw datasets, performs extensive cleaning, and engineers key features like RFM segments, price tiers, and average consumption days.
-   **`EDA.ipynb`**: A comprehensive exploratory data analysis that uncovers the key insights that justify the hybrid model design.
-   **`Apriori Algorithm.ipynb`**: An exploration of association rule mining to find co-purchase patterns, which informed the logic for the complementary slots.
-   **`hybrid_beauty_product_recommender.ipynb`**: The core of the project. This notebook contains the full 5-slot hybrid recommendation engine, from data preparation to final output generation.
-   **`prediction_model.ipynb`**: A separate modeling task to predict the number of days until a customer's next purchase. It compares four regression models and identifies XGBoost as the winner.
-   **`complementary_map.json`**: An AI-generated JSON file that maps 133 Turkish product sub-categories to their logical complements, used by the hybrid recommender.

## Methodology

The project follows a multi-stage methodology:

1.  **Data Engineering**: The raw data is cleaned, and new features are engineered. This includes:
    *   **RFM Segmentation**: Customers are segmented into groups like 'Champions', 'Loyal', and 'Hibernating'.
    *   **Price Tier Inference**: Each customer and product is assigned a price tier (Economy, Mid-Tier, Premium) based on purchase history.
    *   **Proxy Rating Creation**: Since no explicit ratings exist, we create a proxy rating for the SVD model by taking the logarithm of the purchase frequency: `log(1 + purchase_count)`.

2.  **Model Architecture**: The hybrid engine combines three techniques:
    *   **SVD (Collaborative Filtering)**: Used to generate personalized scores for products a user has not yet purchased. This is the backbone for the "New Product Discovery" and "Brand Exploration" slots.
    *   **TF-IDF (Content-Based Filtering)**: Used to find product-line matches for the complementary slots. It calculates the cosine similarity between the names of products (with the brand removed) to find the best match (e.g., `'Abeille Royale Shampoo'` -> `'Abeille Royale Conditioner'`).
    *   **Business Rules**: A strategic layer that enforces diversity (no duplicate categories) and commercial logic (price tier consistency).

3.  **Complementary Map Generation**: A `complementary_map.json` file was generated by providing a list of all 133 unique Turkish product sub-categories to a Large Language Model (GPT-4.1-mini) and prompting it to act as a Turkish beauty expert, creating a logical mapping of which categories complement each other.

## Results

This repository contains two main modeling outputs:

1.  **Hybrid Recommendation System**:
The `hybrid_beauty_product_recommender.ipynb` notebook generates a 5-slot recommendation list for any given member customer. The output is a structured list that balances replenishment, cross-selling, and discovery.

2.  **Next Purchase Day Prediction**:
The `prediction_model.ipynb` notebook trains and evaluates four models to predict the time until a customer's next purchase. The results show that a tuned XGBoost Regressor performs best.

| Model | MAE (Test) | RMSE (Test) | R² (Test) |
| :--- | :--- | :--- | :--- |
| **XGBoost (Tuned)** | **39.04** | **8.18** | **0.375** |
| Gradient Boosting | 39.92 | 8.21 | - |
| Random Forest | 39.65 | 8.22 | - |
| Linear Regression | 55.30 | 8.77 | 0.176 |

## Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/asingh49-cmd/ML_Recommendation_System_Project.git
    cd ML_Recommendation_System_Project
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not currently present but should be created from the notebooks' import statements.)*

### Running the Notebooks

The notebooks should be run in the following order:

1.  `Data Cleaning & Feature Engineering.ipynb`: To generate the clean base dataset.
2.  `EDA.ipynb`: To explore the data and understand the project's rationale.
3.  `hybrid_beauty_product_recommender.ipynb`: To run the main recommendation model.

*Note on Data*: The raw data files are not included in this repository. You will need to update the file paths in the notebooks to point to your local data location.

## Technologies Used

- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Surprise, XGBoost
- **Natural Language Processing**: Scikit-learn (TfidfVectorizer)
- **Other**: Jupyter, OpenAI (for complementary map generation)

## Team

- [Aditya (Adi) Singh](https://github.com/asingh49-cmd)
- [Beste Karnibat](https://github.com/BesteKarnibat)
- [a1b3r](https://github.com/a1b3r)
- [JennyDong021](https://github.com/JennyDong021)
- [JennyLee](https://github.com/jennylee0218)
- [DannyMendoza]

