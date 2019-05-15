side_effects
==============================

Predict DDI and their side effects


## Development, testing, and deployment tools

This section documents our process for running continuous integration (CI) tests and all the steps required during development.  

### Testing

We use nose for all our tests. To run the test, please use:

```bash
pytest -v tests/
```

### Continuous Integration

You should test your code. A rudimentary configuration file for `bitbucket-pipelines.yml` is provided, but do not feel compelled to use it. It is just to help you get started


### Conda Environment:

Please always provide a conda environnment, as the specific package versions you are using are important for your results to be reproductible. A simple test environment file with base dependencies is provided (`env.yml`). Channels are not specified here and therefore respect global Conda configuration

  
### Additional Scripts:

* `scripts`
  * `create_conda_env.py`: Helper program for spinning up new conda environments based on a starter file with Python Version and Env. Name command-line options


## Checklist for updates
- [ ] Make sure there is an/are issue(s) opened for your specific update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR as needed until tests pass
- [ ] Tag the final, debugged version 
   *  `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged in

## Versioneer Auto-version
[Versioneer](https://github.com/warner/python-versioneer) will automatically infer what version 
is installed by looking at the `git` tags and how many commits ahead this version is. The format follows 
[PEP 440](https://www.python.org/dev/peps/pep-0440/) and has the regular expression of:
```regexp
\d+.\d+.\d+(?\+\d+-[a-z0-9]+)
```
If the version of this commit is the same as a `git` tag, the installed version is the same as the tag, 
e.g. `side_effects-0.1.2`, otherwise it will be appended with `+X` where `X` is the number of commits 
ahead from the last tag, and then `-YYYYYY` where the `Y`'s are replaced with the `git` commit hash.


### Copyright

Copyright (c) 2019, InVivo AI

