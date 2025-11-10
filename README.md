# daily-outfit-recommender

Based off of [recommenders](https://github.com/recommenders-team/recommenders)! Their amazing work made this project possible.

## Introduction

Choosing what to wear each day is a common decision that can be surprisingly time-consuming. Existing solutions, like fashion blogs or simple “what to wear” apps, often lack personalization and adaptability. By combining wardrobe tracking with AI-powered recommendations, we aim to create a system that is responsive to the individual user rather than applying one-size-fits-all advice. 

This project builds on a previous attempt at a wardrobe-tracking web app, expanding it into a more intelligent, autonomous recommender system. The focus will not just be on coding but also exploring machine learning, personalization, and user experience design. See previous [project](https://github.com/tealeaf2/loverboy_closet.git).

This is a template meant for research, educational use, personal development of AI recommender systems for outfits. Built for the ML/AI Club at the [University of Notre Dame](https://ai.nd.edu/).

## Getting Started

I recommend that you have these prerequisites before doing anything else, as they will make your life significantly easier.

### Prequisites Required:
- Conda
- Java >= 17+
- Python >= 3.9+
- Github Command Line
- Visual Studio Code

```bash
# 1. Install gcc if it is not installed already. On Ubuntu, this could done by using the command
# sudo apt install gcc

# 2. Create and activate a new conda environment
conda create -n <environment_name> python=3.9
conda activate <environment_name>

# 3. Install the core recommenders package. It can run all the CPU notebooks.
pip install recommenders

# 4. Fork this repository into your own repository using the built in button on Github

# 5. Clone this repo within VSCode or using command line:
git clone <personal_github_link>

# 6. Within VSCode:
#   a. Open the beginning jupyter notebook;  
#   b. Select kernel <environment_name>;
#   c. Run the notebook.
```

### Setup for Spark 

```bash
# 1. Make sure JDK is installed.  For example, OpenJDK 17 can be installed using the command
# sudo apt-get install openjdk-17-jdk

# 2. Adding the spark extra to the pip install command:
pip install recommenders[spark]

# 3. Within VSCode:
#   a. Open a notebook with a Spark model;  
#   b. Select kernel <environment_name>;
#   c. Run the notebook.
```

### Setup for NCF and LightGCN
```bash
# 1. Create a new conda environment (very much recommended) and activate it
# 2. Add the base packages to the environment
pip install recommenders

# 3. Adding the gpu extra to the pip install command:
pip install recommenders[gpu]

# 4. Lower your numpy version for compatability issues (found only on mac so far?):
pip install "numpy<2"

# 5. Within VSCode:
#   a. Open a notebook with a Spark model;  
#   b. Select kernel <environment_name>;
#   c. Run the notebook.
```

## Algorithms

The table below lists the algorithms researched in this project.

| Algorithm | Type | Description |
|-----------|------|-------------|
| Alternating Least Squares (ALS) | Collaborative Filtering | Matrix factorization algorithm for explicit or implicit feedback in large datasets, optimized for scalability and distributed computing capability.
| Neural Collaborative Filtering (NCF) | Collaborative Filtering | Deep learning algorithm with enhanced performance for user/item implicit feedback.
| LightGCN | Collaborative Filtering | Deep learning algorithm which simplifies the design of GCN for predicting implicit feedback. It works in the CPU/GPU environment.
