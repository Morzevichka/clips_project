ENV_NAME = clip_env

PACKAGES = beautifulsoup4 h5py ipython moviepy numpy opencv-python pytubefix selenium torch torchaudio torchvision tqdm

create_env:
	@echo "Creating Conda enviroment: $(ENV_NAME)"
	conda create -y -n $(ENV_NAME) python=3.12.5 pip

install_packages:
	@echo "Installing packages: $(PACKAGES)"
	conda activate $(ENV_NAME) && python -m pip install $(PACKAGES)

	@echo "Installing moviepy from GitHub commit"
	conda activate $(ENV_NAME) && python -m pip install git+https://github.com/Zulko/moviepy.git@bc8d1a831d2d1f61abfdf1779e8df95d523947a5

setup: create_env install_packages
	@echo "Environment $(ENV_NAME) is set up with required packages."

clean:
	@echo "Removing Conda environment: $(ENV_NAME)"
	conda remove -y --name $(ENV_NAME) --all

# make setup