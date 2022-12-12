# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path, PurePath
from dotenv import find_dotenv, load_dotenv
import os

from generate_noise import generate_noise_dataset
from generate_dataset import generate_balanced_dataset


@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filepath=None, output_filepath=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    project_dir = Path(__file__).resolve().parents[2]

    raw = str(project_dir.joinpath(PurePath("data/raw"))) + os.sep
    noise = str(project_dir.joinpath(PurePath("data/noise"))) + os.sep
    processed = str(project_dir.joinpath(PurePath("data/processed"))) + os.sep

    print(noise)

    generate_noise_dataset(path=noise, size = (224,224), path_to_raw=raw, seed=23)
    generate_balanced_dataset(path=processed, size = (224,224), n_copies=1, noise_path=noise, seed=23)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
