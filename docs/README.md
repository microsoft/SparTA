# SparTA Docs

## Local build
This project is hosted by sphinx pipeline and could be accessed publicly in readthedocs. If someone want to make contribution, here is some instructions to preview by local serving.
```python
pip install -r sphinx_requirements.txt
sphinx-autobuild . _build/html
```
Above commands will build and serve the latest (hot updated after modification). Then user could access it in browser by http://127.0.0.1:8000.