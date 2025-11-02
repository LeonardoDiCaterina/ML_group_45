# Cars 4 You: Expediting Car Evaluations with ML

**Course:** Machine Learning 2024/2025
**Project:** Group Project
**Due Date:** December 22nd (17:59)

## 1. Project Context & Business Need

Cars 4 You is an online car resale company. Their current business model requires sending cars to mechanics for evaluation before purchase, which has led to increasing waiting lists and a loss of potential customers to competitors.

The main goal of this project is to address this bottleneck by creating a predictive model. This model will expedite the evaluation process by accurately predicting the price of a car based on details provided by the user, without the immediate need for a mechanic's inspection.

## 2. Project Goals

The project is divided into three main objectives:

1.  **Regression Benchmarking:** Develop and compare multiple regression models to accurately predict car prices (`price`) using a provided dataset. This involves defining a robust model assessment strategy.
2.  **Model Optimization:** Improve the performance of the best-performing models using techniques like hyper-parameter tuning or advanced pre-processing/feature selection.
3.  **Additional Insights (Open-Ended):** An open-ended section to provide further value. Suggestions include:
    * Analyzing feature importance and its contribution to the prediction.
    * Performing an ablation study to measure the contribution of pipeline components.
    * Comparing a single, general model against multiple, specific models (e.g., brand-specific or fuel-type-specific models).

## 3. Dataset

We are provided with two datasets: a training set and a test set.

* **Training Set:** Contains car features and the corresponding ground truth `price` from the company's 2020 database.
* **Test Set:** Contains the same features for an independent set of cars, but without the `price`. Predictions for this set must be submitted to a Kaggle competition.

### Data Attributes

The dataset includes the following attributes:

| Attribute | Description |
| :--- | :--- |
| `carID` | An identifier for each car. |
| `Brand` | The car's main brand (e.g. Ford, Toyota). |
| `model` | The car model. |
| `year` | The year of Registration of the Car. |
| `mileage` | The total reported distance travelled by the car (in miles). |
| `tax` | The amount of road tax (in £) applicable in 2020. |
| `fuelType` | Type of Fuel used by the car (Diesel, Petrol, Hybrid, Electric). |
| `mpg` | Average Miles per Gallon. |
| `engineSize` | Size of Engine in liters (Cubic Decimeters). |
| `paintQuality%` | The mechanic's assessment of the cars' overall paint quality (filled by mechanic). |
| `previousOwners` | Number of previous registered owners of the vehicle. |
| `hasDamage` | Boolean marker filled by the seller stating whether the car is damaged. |
| `price` | **(Target)** The car's price when purchased by Cars 4 You (in £). |

## 4. Project Structure

This project follows the outline specified in the project handout:

1.  **Group Member Contribution**
2.  **Abstract**
3.  **I. Identifying Business Needs**
4.  **II. Data Exploration and Preprocessing**
5.  **III. Regression Benchmarking**
6.  **IV. Open-Ended Section**
7.  **V. Deployment**

*(This repository is organized to follow this structure, with core logic and analysis in the `Notebooks/` directory.)*

## 5. Requirements

To run the notebooks in this project, you will need the libraries listed in `requirements.txt`. Key libraries include:
*(List any other key libraries you ended up using)*

**Note:** The project handout explicitly forbids the use of `XGBoost`, `CatBoost`, `LightGBM`, and `Lazy Predict`. Any use of these packages will result in a penalty.

## 6. How to Run

1.  Clone the repository:
    ```bash
    git clone https://github.com/LeonardoDiCaterina/ML_group_45.git
    cd [YOUR_REPO_NAME]
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open the main notebook to see the full pipeline:
    ```bash
    jupyter notebook Notebooks/Final_ML_template.ipynb
    ```

## 7. Group Members

* [Leonardo Di Caterina]
* [Rodrigo Sardinha]
