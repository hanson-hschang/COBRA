## How to build documentation

External link for documentation will be provided once the repository is public.
The detail implementation of the following `make` commands are in the `Makefile`, and [`SPHINX`](https://www.sphinx-doc.org/en/master/) documnetation tool is used.
For now, use the following commands to build the documentation:
```sh
make clean
make html
```

Once it is build, run the following command to open the document:
```sh
open build/html/index.html
```
