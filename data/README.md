# Dataset 

In this directory, you can add any dataset that your model migh need to work. 
This is therefore a place where non-code related additional information that you want to ship alongside your code (when making it open source)
should go.

Please note that it is not recommended to place large files in your git directory. If your project requires files larger
than a few megabytes in size, please host them elsewhere (Amazon AWS S3). 

## Including data into installation

If you want to include the dataset folder in your package installation (which you should probably not do), modify your package's `setup.py` file and the `setup()` command. 
You will need to include the [`package_data`](http://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use) keyword and point it at the 
correct files.