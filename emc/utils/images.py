"""
Image prediction tools
"""
import numpy as np
import nibabel as nb
from nipype.utils.filemanip import fname_presuffix


def _list_squeeze(in_list):
    return [item[0] for item in in_list]


def flatten(l):
    """
    Flatten list of lists.
    """
    import collections

    for el in l:
        if isinstance(
                el, collections.Iterable) and not isinstance(
                el, (str, bytes)):
            for ell in flatten(el):
                yield ell
        else:
            yield el


def _pass_predicted_outs(ins):
    import os
    from emc.utils.images import flatten

    return [i for i in list(flatten(ins)) if i != '/' and
            os.path.isfile(i) is True]


def mask_4d(dwi_file, mask_file):
    from nipype.utils.filemanip import fname_presuffix

    dwi_masked = fname_presuffix(
        dwi_file,
        use_ext=False,
        suffix="_masked.nii.gz",
    )

    dwi_img = nb.load(dwi_file)
    dwi_data = dwi_img.get_fdata()
    mask_data = nb.load(mask_file).get_fdata()
    data_in_mask = np.nan_to_num(np.broadcast_to(mask_data[..., None],
                                                 dwi_data.shape
                                                 ).astype('bool') * dwi_data)

    nb.Nifti1Image(data_in_mask, affine=dwi_img.affine).to_filename(dwi_masked)
    return dwi_masked


def average_images(images):
    from nilearn.image import mean_img

    average_img = mean_img([nb.load(img) for img in images])
    output_average_image = fname_presuffix(
        images[0], use_ext=False, suffix="_mean.nii.gz"
    )
    average_img.to_filename(output_average_image)
    return output_average_image


def series_files2series_arr(image_list, dtype=np.float32):
    output_array = np.zeros(
        tuple(nb.load(image_list[0]).shape) + (len(image_list),)).astype(dtype=dtype)
    for image_num, image_path in enumerate(image_list):
        output_array[..., image_num] = np.asarray(nb.load(image_path
                                                          ).dataobj).astype(dtype=dtype)
    return output_array


def match_transforms(dwi_files, transforms, b0_indices):
    original_b0_indices = np.array(b0_indices)
    num_dwis = len(dwi_files)
    num_transforms = len(transforms)

    if num_dwis == num_transforms:
        return transforms

    # Do sanity checks
    if not len(transforms) == len(b0_indices):
        raise Exception("Number of transforms does not match number of b0 "
                        "images")

    # Create a list of which emc affines that correspond to the split images
    nearest_affines = []
    for index in range(num_dwis):
        nearest_b0_num = np.argmin(np.abs(index - original_b0_indices))
        this_transform = transforms[nearest_b0_num]
        nearest_affines.append(this_transform)

    return nearest_affines


def save_4d_to_3d(in_file):
    in_img = nb.load(in_file)
    if len(in_img.shape) > 3 or (len(in_img.shape) == 4 and
                                 in_img.shape[-1] == 1):
        files_3d = nb.four_to_three(in_img)
        out_files = []
        for i, file_3d in enumerate(files_3d):
            out_file = fname_presuffix(in_file, suffix="_tmp_{}".format(i))
            file_3d.to_filename(out_file)
            out_files.append(out_file)
        del files_3d
    else:
        out_file = fname_presuffix(in_file, suffix="_tmp_{}".format(0))
        in_img.to_filename(out_file)
        out_files = [out_file]
    in_img.uncache()

    return out_files


def prune_b0s_from_dwis(in_files, b0_ixs):
    """
    Remove *b0* volume files from a complete list of DWI volume files.

    Parameters
    ----------
    in_files : list
        A list of NIfTI file paths corresponding to each 3D volume of a
        DWI image (i.e. including B0's).
    b0_ixs : list
        List of B0 indices.

    Returns
    -------
    out_files : list
       A list of file paths to 3d NIFTI images.

    Examples
    --------
    >>> os.chdir(tmpdir)
    >>> b0_ixs = np.where(np.loadtxt(str(dipy_datadir / "HARDI193.bval")) <= 50)[0].tolist()[:2]
    >>> in_file = str(dipy_datadir / "HARDI193.nii.gz")
    >>> threeD_files = save_4d_to_3d(in_file)
    >>> out_files = prune_b0s_from_dwis(threeD_files, b0_ixs)
    >>> assert sum([os.path.isfile(i) for i in out_files]) == len(out_files)
    >>> assert len(out_files) == len(threeD_files) - len(b0_ixs)
    """
    if in_files[0].endswith("_warped.nii.gz"):
        out_files = [
            i
            for j, i in enumerate(
                sorted(
                    in_files, key=lambda x: int(x.split("_")[-2].split(".nii.gz")[0])
                )
            )
            if j not in b0_ixs
        ]
    else:
        out_files = [
            i
            for j, i in enumerate(
                sorted(
                    in_files, key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
                )
            )
            if j not in b0_ixs
        ]
    return out_files


def save_3d_to_4d(in_files):
    img_4d = nb.funcs.concat_images([nb.load(img_3d) for img_3d in in_files])
    out_file = fname_presuffix(in_files[0], suffix="_merged")
    img_4d.to_filename(out_file)
    del img_4d
    return out_file


def get_params(A):
    """This is a copy of spm's spm_imatrix where
    we already know the rotations and translations matrix,
    shears and zooms (as outputs from fsl FLIRT/avscale)
    Let A = the 4x4 rotation and translation matrix
    R = [          c5*c6,           c5*s6, s5]
        [-s4*s5*c6-c4*s6, -s4*s5*s6+c4*c6, s4*c5]
        [-c4*s5*c6+s4*s6, -c4*s5*s6-s4*c6, c4*c5]
    """

    def rang(b):
        a = min(max(b, -1), 1)
        return a

    Ry = np.arcsin(A[0, 2])
    # Rx = np.arcsin(A[1, 2] / np.cos(Ry))
    # Rz = np.arccos(A[0, 1] / np.sin(Ry))

    if (abs(Ry) - np.pi / 2) ** 2 < 1e-9:
        Rx = 0
        Rz = np.arctan2(-rang(A[1, 0]), rang(-A[2, 0] / A[0, 2]))
    else:
        c = np.cos(Ry)
        Rx = np.arctan2(rang(A[1, 2] / c), rang(A[2, 2] / c))
        Rz = np.arctan2(rang(A[0, 1] / c), rang(A[0, 0] / c))

    rotations = [Rx, Ry, Rz]
    translations = [A[0, 3], A[1, 3], A[2, 3]]

    return rotations, translations

