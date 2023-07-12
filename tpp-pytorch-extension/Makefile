
format:
	find src -name "*.cpp" -or -name "*.h"  |  xargs -t -n 1 clang-format -i -style=file
	black src/tpp_pytorch_extension examples setup.py

install:
	python setup.py install

reinstall:
	pip uninstall -y tpp-pytorch-extension
	python setup.py clean
	rm -rf build dist
	python setup.py install

clean:
	python setup.py clean
