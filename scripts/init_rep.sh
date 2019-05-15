pip install click
conda env export | grep -v "^prefix:" > conda_env.yml
ls -rtd1 * | grep -v 'scripts'  | sed 's/^/..\/..\//' | grep -v 'pycache' > scripts/sagemaker/contents.txt