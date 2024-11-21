### INSTALLATION

1. Create conda enviroment
```bash
conda create -n {NAME_ENV}
```

2. Activate your enviroment

```bash
conda activate {NAME_ENV}
```

3. Install python and pip into your project

```bash
conda install python=3.12.5 pip
```

4. Install requirements packages

```bash
python -m pip -r /path/to/requirements.txt
```

5. Also install moviepy dev
   
```bash
python -m pip install git+https://github.com/Zulko/moviepy.git@bc8d1a831d2d1f61abfdf1779e8df95d523947a5
```

6. Choose {NAME_ENV} environment in your IDE to work with