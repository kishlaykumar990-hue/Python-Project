# Ideal-Function-Fitter

## 📌 Project Overview

This repository contains my university project for Semester 1, focused on working with datasets using Python.
The goal is to:

* Load training, ideal, and test datasets.

* Find the four ideal functions with the smallest deviation from the training data, using least squares (SSE).

* Map the test dataset to those functions, using a sqrt(2) deviation threshold.

* Store results in tables & a SQLite database.

* Generate visualizations for analysis.

The project demonstrates concepts of data handling, deviation analysis, database usage, exception handling, unit testing, and visualization in Python.

## 📂 Project Structure
```

│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── ideal.csv
│
├── docs/
│   └── Task_WrittenAssignment_DLMDSPWP01.pdf
|
├── src/
│   ├── main.py
│   ├── train_loader.py
│   ├── ideal_loader.py
│   ├── test_loader.py
│   ├── deviation_calculator.py
│   ├── test_handler.py
│   ├── exceptions.py
│   └── __init__.py
│
├── unit_test/
│   ├── test_pipeline.py
│   ├── run_tests.py
│   ├── fixtures/
│   │   └── fixture_ideal.csv
│   └── test_results/
│       └── Test_Results.txt
│
├── outputs/
│   ├── test_vs_ideal.html
│   ├── train_vs_ideal.html
│   ├── final_table_preview.csv
│   ├── function_mapping.db
│   ├── test_deviations_preview.csv
│   ├── train_deviations_preview.csv
│
├── LICENSE
├── requirements.txt
└── README.md
```



## 📝 Problem Statement

Detailed instructions are available in the assignment file:
[Task_WrittenAssignment_DLMDSPWP01.pdf](./docs/Task_WrittenAssignment_DLMDSPWP01.pdf)



## 🚀 How It Works

1. Load Data

    * Training, ideal, and test datasets are read directly from `.csv` files (`train_loader.py`, `ideal_loader.py`, `test_loader.py`).

    * Raw training and ideal data are stored into SQLite tables.

2. Find Ideal Functions

    * `deviation_calculator.py` compares each training function against all 50 ideal functions and selects the one with the lowest sum of squared errors (SSE), computed live at runtime.

3. Test Data Handling

    * Test dataset points are interpolated against the selected ideal functions and checked against a sqrt(2) deviation threshold (computed live from the training run).

    * If no test points can be mapped at all, a custom exception (`MappingErrorHandling`) is raised.

    * Deviations and the final mapping are stored in CSV/DB.

4. Visualization

   Two interactive Bokeh charts are generated: `test_vs_ideal.html` (ideal functions + assigned test points) and `train_vs_ideal.html` (raw training data + matched ideal functions), each split into a 2x2 grid (one panel per function) for readability.

5. Unit Tests

   `unit_test/test_pipeline.py` covers the selection logic, threshold logic, final-table mapping, the custom exception, both loaders, and a database round-trip, using small fabricated data.


## ⚙️ Requirements

Make sure you have the following installed:
```
   pandas
   numpy
   bokeh
   sqlalchemy
   pytest
```

## ▶️ Usage

   1. Quick Start

```
     # Clone repository
     git clone https://github.com/kishlaykumar990-hue/Python-Project.git

     # Install dependencies
     pip install -r requirements.txt
```
   2. Run the program (from the project root folder)
   ```
     python3 -m src.main
```
   3. Run the unit tests
   ```
     python3 unit_test/run_tests.py
```

   4. Check outputs

       * Mappings & deviations in `.db` and `.csv` files inside `outputs/`.

       * Interactive visualizations: `outputs/test_vs_ideal.html` and `outputs/train_vs_ideal.html`.


## 📈 Results

  The analysis selects 4 optimal ideal functions and maps test data points:
  ```
    Selected Functions (via least squares / SSE):
    y1 -> Y13 (SSE ~34.08)
    y2 -> Y24 (SSE ~33.45)
    y3 -> Y36 (SSE ~35.57)
    y4 -> Y40 (SSE ~35.00)
  ```


## 📜 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.

## 👨‍💻 Author

[Kishlay Kumar](https://github.com/kishlaykumar990-hue)
