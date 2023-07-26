from setuptools import setup, find_packages


setup(
    name="blueprint-trainer",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    license="MIT",
    description="Pytorch Trainer with Blueprint",
    author="Jianbin Chang",
    author_email="shjwudp@gmail.com",
    long_description_content_type = 'text/markdown',
    url = "https://github.com/shjwudp/blueprint-trainer",
    keywords = [
        "artificial intelligence",
        "deep learning",
        "training",
    ],
    install_requires=[
        "torch",
        "tabulate",
        "plotext",
        "datasets",
        "transformers",
        "wandb",
        "omegaconf",
    ],
)
