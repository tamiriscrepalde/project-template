# {{cookiecutter.project_name}}


- [{{cookiecutter.project_name}}](#{{cookiecutter.project_name_ref}})
	- [Context](#context)
	- [Files in the repository](#files-in-the-repository)
	- [Setup instructions](#setup-instructions)


## Context

{{cookiecutter.context}}


## Files in the repository

Repository structure:

- notebooks
  - exploratory_data_analysis
    - exploratory_data_analysis.ipynb
    - README.md
  - experimentation
- src
  - visualization
    - visualization_utils.py
  - utils.py
- .gitignore
- GoogleUtils.py
- README.md
- requirements.txt


## Setup instructions

1. Clone this repository:
   `git clone {{cookiecutter.repository_to_clone}}`.

2. Build the dev container. 

3. Install the required libraries by running the command: `pip install -r requirements.txt`.

4. Run the notebook.
