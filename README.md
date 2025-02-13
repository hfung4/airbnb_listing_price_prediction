<h1 align="center">
Airbnb listing price predcition 

## Set up your environment
I use Databricks 15.4 LTS runtime, which requires Python 3.11.

I used uv as virtual environment manager, to create a new environment and create a lockfile, run:

```
uv venv -p 3.11 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```
