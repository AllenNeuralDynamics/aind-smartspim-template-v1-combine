"""
Combines registered brain volumes into a single
image creating a template
"""

import argparse
import glob
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ants
import matplotlib.pyplot as plt
import numpy as np
from _shared.types import PathLike
from ants import iMath

TGT_NUM_IMGS = 20  # TODO


def create_logger(output_log_path: PathLike, prefix: str = "initial") -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    prefix: str
        Prefix in the log filename

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/{prefix}_average_template.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info(f"Execution datetime: {CURR_DATE_TIME}")

    return logger


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def combine_images(image_list: List[str], logger: logging.Logger):
    """
    Combine a list of images by averaging them with weights.

    Parameters
    ----------
    image_list : List[str]
        List of paths to the images to be combined.
    logger : logging.Logger
        Logger to record information during processing.

    Returns
    -------
    ants.image
        The combined average image.
    """
    weights = np.repeat(1.0 / len(image_list), len(image_list))
    weights = [x / sum(weights) for x in weights]

    logger.info(f"Weights for {len(image_list)} images: {weights}")

    xavg = None
    for i in range(len(image_list)):
        logger.info(f"Adding {image_list[i]} to the template...")
        image = ants.image_read(str(image_list[i]))
        if i == 0:
            xavg = image * 0
        temp = image * weights[i]
        # temp = resample_image_to_target(temp, xavg)
        xavg = xavg + temp

    return xavg


def compute_median_image(images_list: List[str], logger: logging.Logger):
    """
    Compute the median image from a list of images.

    Parameters
    ----------
    images_list : List[str]
        List of paths to the images.
    logger : logging.Logger
        Logger to record information during processing.

    Returns
    -------
    ants.image
        The computed median image.
    """
    # TODO: avoid OOM
    image = ants.image_read(str(images_list[0]))

    # creat an empty array
    combine_imgs = np.zeros((len(images_list), *(image.shape)))
    logger.info(f"Combine_imgs shape: {combine_imgs.shape}")

    # add first image
    logger.info(f"Adding {images_list[0]} to the template...")
    combine_imgs[0, ...] = image.numpy().astype("float32")

    # add other images
    for i in range(1, len(images_list)):
        logger.info(f"Adding {images_list[i]} to the template...")
        combine_imgs[i, ...] = (
            (ants.image_read(str(images_list[i]))).numpy().astype("float32")
        )

    logger.info("Finish adding images")

    # compute median
    logger.info(f"Compute median ......")
    start_time = time.time()
    median_img = np.median(combine_imgs.astype("float32"), axis=0)
    logger.info(f"Median image shape: {median_img.shape}")

    ants_img = ants.from_numpy(
        median_img.astype("float32"),
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
    )

    end_time = time.time()
    logger.info(f"Compute median image execution time: {end_time-start_time}")

    return ants_img


def morphology_update(
    image, affine_list, warp_list, gradient_step, avg_affine_path, avg_warp_path, logger
):
    """
    Update the image using morphology based on affine and warp transformations.

    Parameters
    ----------
    image : ants.image
        The input image to be updated.
    affine_list : List
        List of affine transformation paths.
    warp_list : List
        List of warp transformation paths.
    gradient_step : float
        Step used for the update process.
    avg_affine_path : str
        Path to save the average affine transformation.
    avg_warp_path : str
        Path to save the average warp transformation.
    logger : logging.Logger
        Logger to record information during processing.

    Returns
    -------
    ants.image
        The updated image after applying transformations.
    """
    avg_affine = ants.average_affine_transform(affine_list)
    ants.write_transform(avg_affine, avg_affine_path)

    if len(warp_list):
        average_warp_delta = (
            combine_images(warp_list, logger) * (-1.0) * gradient_step
        )  # TODO, questions what does average_warp_delta for?

        # apply affine to the nonlinear?
        # need to save the average
        wavgA = ants.apply_transforms(
            fixed=image,
            moving=average_warp_delta,
            imagetype=1,
            transformlist=avg_affine_path,
            whichtoinvert=[1],
        )

        ants.image_write(wavgA, avg_warp_path)

        return ants.apply_transforms(
            fixed=image,
            moving=image,
            transformlist=[avg_warp_path, avg_affine_path],
            whichtoinvert=[0, 1],
        )
    else:
        return ants.apply_transforms(
            fixed=image,
            moving=image,
            transformlist=[avg_affine_path],
            whichtoinvert=[1],
        )


def sharpen_image(image, blending_weight=0.75):
    """
    Sharpen the input image using a blending technique.

    Parameters
    ----------
    image : ants.image
        The input image to be sharpened.
    blending_weight : float, optional
        Weight for blending the sharpened image (default is 0.75).

    Returns
    -------
    ants.image
        The sharpened image.
    """
    return image * blending_weight + iMath(image, "Sharpen") * (1.0 - blending_weight)


def plot_antsimgs_1(ants_reg, figpath, title, vmin=0, vmax=500):
    """
    Plot 3 slices of the 3D ANTs image and save it to a file.

    Parameters
    ----------
    ants_reg : ants.image
        The 3D ANTs image to be plotted.
    figpath : str
        Path to save the figure.
    title : str
        Title for the figure.
    vmin : float, optional
        Minimum value for color scaling (default is 0).
    vmax : float, optional
        Maximum value for color scaling (default is 500).
    """
    if figpath:
        half_size = np.array(ants_reg.shape) // 2
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        ax[0].imshow(ants_reg[half_size[0], :, :], cmap="gray", vmin=vmin, vmax=vmax)
        ax[1].imshow(ants_reg[:, half_size[1], :], cmap="gray", vmin=vmin, vmax=vmax)
        im = ax[2].imshow(
            ants_reg[
                :,
                :,
                half_size[2],
            ],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        fig.suptitle(title, y=0.8)
        # fig.show()

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)

        plt.colorbar(im, ax=ax.ravel().tolist(), fraction=0.1, shrink=0.7)
        plt.savefig(figpath)


def plot_antsimgs(ants_img, figpath, title, vmin=0, vmax=500):
    """
    Plot the 3D ANTs image and save it to a file.

    Parameters
    ----------
    ants_img : ants.image
        The 3D ANTs image to be plotted.
    figpath : str
        Path to save the figure.
    title : str
        Title for the figure.
    vmin : float, optional
        Minimum value for color scaling (default is 0).
    vmax : float, optional
        Maximum value for color scaling (default is 500).
    """
    # plot_antsimgs_0(ants_img.numpy(), f"{figpath}_0.jpg", f"{title}_0", vmin=vmin, vmax=vmax)
    plot_antsimgs_1(
        ants_img.numpy(), f"{figpath}_1.jpg", f"{title}_1", vmin=vmin, vmax=vmax
    )
    # plot_antsimgs_2(ants_img.numpy(), f"{figpath}_2.jpg", f"{title}_2", vmin=vmin, vmax=vmax)


def verify_number_images(actual_num: int, expected_num: tuple) -> None:
    """Verify the number of images"""
    if actual_num != expected_num:
        raise ValueError(
            f"The number of images is {actual_num}, not euqal to the expected number {expected_num}!"
        )


def create_initial_average_template(
    registered_paths: List[str],
    output_path: PathLike,
    gradient_step: Optional[float] = 0.25,
):
    """
    Creates an initial average template. This average is voxel-wise.

    Parameters
    ----------
    registered_paths: List[str]
        Paths of the registered brains to the
        CCF Allen atlas.

    output_path: PathLike
        Path where the initial average template
        will be saved.

    gradient_step: Optional[float]
        Step used to move to the average image.
        Lower value means shorter jumps and
        a better result to the mean image.

    """
    registered_brains = {}

    # Declaring paths
    output_initial_average_template = f"{output_path}/initial_template/"
    create_folder(dest_dir=output_initial_average_template, verbose=True)

    logger = create_logger(output_log_path=output_initial_average_template)

    logger.info(f"Found brains: {registered_paths}")

    logger.info(f"Running creation of initial template")

    # -----------------------------------#
    # get image paths
    # -----------------------------------#

    logger.info("Getting registered brains")
    for registered_path in registered_paths:
        registered_path = Path(registered_path)
        registered_image = registered_path.joinpath("moving.nii.gz")
        affine_mat = registered_path.joinpath("0GenericAffine.mat")
        warp_mat = registered_path.joinpath(
            "1Warp.nii.gz"
        )  # TODO: question where is 1Warp.nii.gz??, can not find it in preprocess step

        brain_name = registered_path.name

        logging.info(f"Loading {registered_image} for brain id {brain_name}")

        registered_brains[brain_name] = {}
        if os.path.exists(registered_image):
            registered_brains[brain_name]["registered_image"] = registered_image

        if os.path.exists(affine_mat):
            registered_brains[brain_name]["affine_mat"] = affine_mat

        if os.path.exists(warp_mat):
            registered_brains[brain_name]["warp_mat"] = warp_mat

    output_image_path = Path(f"{output_initial_average_template}fixed.nii.gz")
    figure_path = Path(f"{output_initial_average_template}fixed.jpg")

    logger.info(f"Collected data: {registered_brains}")

    # -----------------------------------#
    # collect the images
    # -----------------------------------#

    images_list = []
    affine_list = []
    warp_list = []

    for brain_id, values in registered_brains.items():
        registered_path = values.get("registered_image")
        affine_path = values.get("affine_mat")
        warp_path = values.get("warp_mat")

        if registered_path:
            images_list.append(registered_path)
        else:
            logger.warning(f"Please, check registered path in dataset {brain_id}")

        if affine_path:
            affine_list.append(affine_path)
        else:
            logger.warning(f"Please, check affine path in dataset {brain_id}")

        if warp_path:
            warp_list.append(warp_path)
        else:
            logger.warning(f"Please, check warp path in dataset {brain_id}")

    logger.info(f"Images: {images_list}")
    logger.info(f"Affines: {affine_list}")
    logger.info(f"Warps: {warp_list}")

    logger.info(f"** Number of Images: {len(images_list)} **")
    logger.info(f"** Number of Affines: {len(affine_list)} **")
    logger.info(f"** Number of Warps: {len(warp_list)} **")

    # check if the number of images is equal to 20 before computing median
    verify_number_images(len(images_list), TGT_NUM_IMGS)

    # -----------------------------------------------#
    # combine registered images, moving.nii.gz
    # -----------------------------------------------#

    logger.info(f"Combining images with: {images_list}")
    start_time = time.time()
    # image = combine_images(image_list=images_list, logger=logger)
    image = compute_median_image(images_list=images_list, logger=logger)

    end_time = time.time()

    logging.info(f"Finishing combination, time {end_time - start_time} s")

    logger.info(f"Writing initial average template in: {output_image_path}")
    ants.image_write(image, str(output_image_path))

    plot_antsimgs(
        image, f"{output_initial_average_template}fixed", "fixed1", vmin=0, vmax=1.5
    )


def update_average_template(
    round_step: str,
    registered_paths: List[str],
    output_path: PathLike,
    gradient_step: Optional[float] = 0.25,
):
    """
    Update the average template image by combining registered brain images
    and applying transformations.

    Parameters
    ----------
    round_step : str
        A string indicating the current round of template updating.
    registered_paths : List[str]
        A list of file paths pointing to the directories
        containing registered brain images.
    output_path : PathLike
        The directory where the updated average template will be saved.
    gradient_step : float, optional
        A step size for gradient updates during morphology
        processing (default is 0.25).

    Raises
    ------
    FileNotFoundError
        If the registered image, affine transformation, or warp
        transformation files are not found.

    ValueError
        If the number of images or affines does not match the
        target number defined by TGT_NUM_IMGS.

    Notes
    -----
    - The function first creates a folder to save the updated
    average template and initializes a logger.
    - It collects paths to registered images, affine transformations,
    and warp transformations.
    - The function computes the median image from the collected images.
    - Morphology updates are applied if affine or warp transformations are provided.
    - The final updated average template is saved as both a NIfTI image and a plot.
    """
    registered_brains = {}

    # Declaring paths
    output_updated_average_template = (
        f"{output_path}/update_{round_step}_average_template/"
    )
    create_folder(dest_dir=output_updated_average_template, verbose=True)

    logger = create_logger(output_log_path=output_updated_average_template)
    logger.info(f"Found brains: {registered_paths}")

    logger.info(f"Running template update: {round_step}")

    # -----------------------------------#
    # get image paths
    # -----------------------------------#

    logger.info("Getting registered brains")
    for registered_path in registered_paths:
        registered_path = Path(registered_path)
        registered_image = registered_path.joinpath("moving.nii.gz")
        affine_mat = registered_path.joinpath("0GenericAffine.mat")
        warp_mat = registered_path.joinpath("1Warp.nii.gz")

        brain_name = registered_path.name

        logging.info(f"Loading {registered_image} for brain id {brain_name}")

        registered_brains[brain_name] = {}
        if os.path.exists(registered_image):
            registered_brains[brain_name]["registered_image"] = registered_image

        if os.path.exists(affine_mat):
            registered_brains[brain_name]["affine_mat"] = affine_mat

        if os.path.exists(warp_mat):
            registered_brains[brain_name]["warp_mat"] = warp_mat

    output_image_path = Path(f"{output_updated_average_template}fixed.nii.gz")
    figure_path = Path(f"{output_updated_average_template}fixed.jpg")

    # -----------------------------------#
    # collect the images
    # -----------------------------------#

    logger.info(f"Collected data: {registered_brains}")

    images_list = []
    affine_list = []
    warp_list = []

    for brain_id, values in registered_brains.items():
        registered_path = values.get("registered_image")
        affine_path = values.get("affine_mat")
        warp_path = values.get("warp_mat")

        if registered_path:
            images_list.append(registered_path)
        else:
            logger.warning(f"Please, check registered path in dataset {brain_id}")

        if affine_path:
            affine_list.append(affine_path)
        else:
            logger.warning(f"Please, check affine path in dataset {brain_id}")

        if warp_path:
            warp_list.append(warp_path)
        else:
            logger.warning(f"Please, check warp path in dataset {brain_id}")

    logger.info(f"Images: {images_list}")
    logger.info(f"Affines: {affine_list}")
    logger.info(f"Warps: {warp_list}")

    logger.info(f"** Number of Images: {len(images_list)} **")
    logger.info(f"** Number of Affines: {len(affine_list)} **")
    logger.info(f"** Number of Warps: {len(warp_list)} **")

    # TODO: check if the number of images is equal to 20 before computing median
    verify_number_images(len(images_list), TGT_NUM_IMGS)
    verify_number_images(len(affine_list), TGT_NUM_IMGS)

    # -----------------------------------#
    # compute median
    # -----------------------------------#

    logger.info(f"Combining images with: {images_list}")
    start_time = time.time()
    # image = combine_images(image_list=images_list, logger=logger) # compute the mean
    image = compute_median_image(
        image_list=images_list, logger=logger
    )  # compute the median
    end_time = time.time()

    logger.info(
        f"[{round_step}] Writing updated average template in: {output_image_path}"
    )

    plot_antsimgs(
        image,
        f"{output_updated_average_template}fixed_median",
        "fixed_median",
        vmin=0,
        vmax=1.5,
    )
    ants.image_write(
        image, str(Path(f"{output_updated_average_template}fixed_median.nii.gz"))
    )

    # -----------------------------------#
    #  morphology update
    # -----------------------------------#

    if list(affine_list) or list(warp_list):
        logging.info("Applying morphology update")
        start_time = time.time()
        image = morphology_update(
            image=image,
            affine_list=affine_list,
            warp_list=warp_list,
            gradient_step=gradient_step,
            avg_affine_path=f"{output_updated_average_template}average_affine.mat",
            avg_warp_path=f"{output_updated_average_template}average_warp.nii.gz",
            logger=logger,
        )
        end_time = time.time()

        logger.info("Laplacian shapening")
        start_time = time.time()
        image = sharpen_image(image)
        end_time = time.time()

    logging.info(f"Finishing combination, time {end_time - start_time} s")

    logger.info(
        f"[{round_step}] Writing updated average template in: {output_image_path}"
    )

    ants.image_write(image, str(output_image_path))
    plot_antsimgs(
        image, f"{output_updated_average_template}fixed", "fixed", vmin=0, vmax=1.5
    )


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default="initial", nargs="?")
    parser.add_argument("gradient_step", default="0.25", nargs="?")
    parser.add_argument("round_step", default="round_1", nargs="?")

    args = parser.parse_args()

    mode = str(args.mode)
    gradient_step = float(args.gradient_step)
    round_step = str(args.round_step)

    RESULTS_FOLDER = os.path.abspath("../results")
    DATA_FOLDER = os.path.abspath("../data")  # TODO

    # DATA_FOLDER = os.path.abspath("../data/ccf_10um_template_percNorm_round0/preprocess") # TODO

    # DATA_FOLDER = os.path.abspath("../data/ccf_template_round0_10brains_resampled_zscore_median/preprocess_resampled_zscored_image")

    registered_paths = glob.glob(f"{DATA_FOLDER}/*/")
    print("Data folder content: ", os.listdir(DATA_FOLDER))

    if len(registered_paths) > 1:
        if mode == "initial":
            create_initial_average_template(
                registered_paths=registered_paths,
                output_path=RESULTS_FOLDER,
                gradient_step=gradient_step,
            )

        elif mode == "update":
            update_average_template(
                round_step=round_step,
                registered_paths=registered_paths,
                output_path=RESULTS_FOLDER,
                gradient_step=gradient_step,
            )

        else:
            raise ValueError(f"Invalid mode: {mode}")

    else:
        raise ValueError(
            f"Please, check the number of brains to average. Read brains: {registered_paths}"
        )


if __name__ == "__main__":
    main()
