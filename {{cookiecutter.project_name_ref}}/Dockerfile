FROM python:3.11.9

WORKDIR /{{cookiecutter.project_name_ref}}

RUN pip install poetry

COPY pyproject.toml poetry.lock* /{{cookiecutter.project_name_ref}}/

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY . /{{cookiecutter.project_name_ref}}

RUN poetry run python -m ipykernel install --user --name={{cookiecutter.project_name_ref}}-env --display-name "{{cookiecutter.project_name_ref}}-env"
