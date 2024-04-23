# Project Template


- [Project Template](#project-template)
  - [Context](#context)
  - [Files in the repository](#files-in-the-repository)


## Context

This repository aims to version a project template in the context of machine learning projects.

It is a work in progress. My goal is to update the template over time to include processes, tools and libs that can help in the development of data science projects, especially in terms of development time.


## Files in the repository

Repository structure:

- {{cookiecutter.project_name_ref}}
  - .devcontainer
    - devcontainer.json
  - notebooks
    - exploratory_data_analysis.ipynb
    - modeling.ipynb
  - src
    - visualization
      - visualization_utils.py
    - utils.py
  - .gitignore
  - Dockerfile
  - pyproject.toml
  - README.md
- cookiecutter.json
- README.md
