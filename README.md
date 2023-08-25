# Customer Personality Analysis

# Final Proyect

This repository contains the code and resources required to implement a customer personality analysis and clustering system using MLOps techniques. The primary goal of the project is to apply processes and best practices for managing the development, testing, and deployment workflow of machine learning models.

## Kaggle link

<https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis>

### **Problem Statement**

Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.

**Target** is: to perform clustering to summarize customer segments.

**2. What solutions (notebooks) have you already developed?**

There are already many notebooks created around this problem. Some are exploring new ways of clustering and others are dedicated to further understanding the problem.

This time I will take the following notebooks as a base:

    https://www.kaggle.com/code/gaganmaahi224/9-clustering-techniques-for-customer-segmentation#Evaluating-models

    https://www.kaggle.com/code/kslarwtf/eda-clustering-updated

    https://www.kaggle.com/code/alisultanov/clustering-customer-personality-analysis#Features:

**3. Which of all the solutions contains the minimum necessary to be able to train and save a model?**

        https://www.kaggle.com/code/gaganmaahi224/9-clustering-techniques-for-customer-segmentation#Evaluating-models

**4. Define the objective of the project, in this case it is only for the subject and it is a proof of concept.**

The objective of this project is to use all the MLOps tools seen in the course to:

    1. Deploy a tool that can identify the Customer segment.
    2. Deploy a tool that can do training on a new dataset.

**5. What models can be trained beyond the solution already created in Kaggle?**

There are already so many solutions in Kaggle, it is difficult to find which ones have not been used. What is certain is that for this exercise we will limit ourselves to not using excessively complex solutions, to facilitate the deployment and achieve adequate times in execution and according to the program.

**6. What will be the end result of this project?**

A tool (API) that can do training on a new dataset and identify the Customer segment for one person.

## Pytest (Unit test)

You can find the tests in the [tests](tests) folder.

### Virtual environment

1. Create a virtual environment with `Python 3.10+`
    * Create venv

        ```bash
        python3.10 -m venv venv-tests
        ```

    * Activate the virtual environment

        ```
        for linux: 
        source venv-tests/bin/activate
        
        for windows: 
        ..\.venv\Scripts\activate
        ```

    * Install the packages

        ```bash
        pip install requirements.txt
        ```

    > **NOTE:**
    Deactivate the virtual environment using this command at the end of its example.  

        deactivate

**The configuration is ready for the check the script!**

### Testing Data Processing and Analysis

This repository contains a set of test scripts for verifying the correctness of data processing and analysis functions using the Python programming language. These tests ensure that the implemented functions behave as expected and produce the desired outcomes. The tests are organized into separate files for each specific area of functionality.

Test Scripts

The repository includes the following test scripts:

    1. *test_data_existence.py*

This script tests the existence and validity of the input data. It checks whether the required datasets or files are present and properly formatted.

    2.*test_feature_generation.py*: 

This script tests the feature generation function. It ensures that the new features generated are correct and meet the intended specifications.

    3. *test_model_existence.py*: 

This script tests the existence of trained models. It verifies whether the saved models exist in the designated directory and can be loaded successfully.

    4. *test_remove_outliers.py*:     
This script tests the outlier removal function. It validates whether the outliers are properly identified and removed, resulting in a clean dataset.

#### Running the Tests

To run the tests, follow these steps:

Make sure you have Python installed on your system.

Install the required dependencies using the following command:

```
    pip install -r requirements.txt
```

Run the desired test script using the following command:

    python test_script_name.py

*Replace test_script_name.py with the actual name of the test script you want to run (e.g., test_data_existence.py).*

The test script will execute and provide output indicating whether the tests have passed or failed.

### pre-commit Configuration

The `pre-commit` system allows you to automate code review and formatting tasks before committing to your repository. This ensures that the code meets certain standards and quality before being recorded in the repository's history.

#### Steps to Set Up pre-commit in Your Project

1. **Installation**: Make sure you have `pre-commit` installed. If you don't, you can install it using pip:

   ```sh
   pip install pre-commit

**Additional Resources**

    pre-commit Repository: 
        https://github.com/pre-commit/pre-commit

        List of pre-commit Compatible Hooks: 
        https://pre-commit.com/hooks.html
        
        isort Documentation: 
        https://github.com/pre-commit/mirrors-isort
        
        autoflake Documentation: 
        https://github.com/pre-commit/mirrors-autoflake
        
        autopep8 Documentation: 
        https://github.com/pre-commit/mirrors-autopep8
