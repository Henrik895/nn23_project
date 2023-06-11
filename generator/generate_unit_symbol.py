import numpy as np
import cv2
from random import randint
import random
from typing import Tuple, List, Dict, Optional


def get_random(label: str,
               sample: Dict[str, List[np.ndarray]]) -> np.ndarray:
    """
    Chooses random image from the sample list given the label.

    Args:
        label : The label from which we wish to sample.
        sample : The dictionary containing the images.

    Return:
        Sampled image which corresponds to label.
    """
    imgs = sample[label]
    return np.copy(imgs[randint(0, len(imgs)-1)])


def place_symbol(canvas: np.ndarray,
                 symbol: np.ndarray,
                 point1: int,
                 point2: int) -> np.ndarray:
    """
    Places the given symbol on canvas. Only places the black pixels.

    Args:
        canvas : The image where the symbol will be placed
        symbol : The image which will be placed on template
        point1 : The upper edge on template where the symbol will be placed.
        point2 : The left edge on template where the symbol will be placed.

    Returns:
        canvas: A image with placed symbol.
    """

    canvas[point1:point1+symbol.shape[0],
           point2:point2+symbol.shape[1]][symbol < 140] = 0
    return canvas


def resize_image(template: np.ndarray,
                 unit_img: np.ndarray,
                 unit_lab: str,
                 resizable: List[str],
                 resizable_horizontal: List[str],
                 resizable_vertical: List[str]) -> np.ndarray:
    """
    Given the template and the unit image resizes the unit images with respect to template.
    The images are reshaped with respect to template shape and in theory should work with every input image.
    However, this function is only tested with on example images where there was one image per example
    and due to that the porportions may work for only those.

    Args:
        template : The specific images from which we wish to sample.
        unit_img : The dictionary containing the images.
        unit_lab : Label of the unit_img
        resizable : These labels are resized to template shape
        resizable_horizontal : These images width is equal to template, but the height is not.
        resizable_vertical : These images height is equal to template, but the width is not.

    Return:
        Return the new image which is with same shape as template and on the new image the given unit_img is placed into right positon.
    """
    unit_img2 = np.full(template.shape, 255)
    if unit_lab in resizable:
        unit_img = cv2.resize(unit_img, [template.shape[1], template.shape[0]])
        unit_img2 = unit_img
    elif unit_lab in resizable_horizontal:
        unit_img = cv2.resize(unit_img, [template.shape[1], unit_img.shape[0]])
        if unit_lab == "hq_unit":
            unit_img2 = place_symbol(
                unit_img2, unit_img, int(unit_img2.shape[0]*0.2), 0)
        else:
            unit_img2 = place_symbol(
                unit_img2, unit_img, int(unit_img2.shape[0]*0.7), 0)
    elif unit_lab in resizable_vertical:
        unit_img = cv2.resize(unit_img, [unit_img.shape[1], template.shape[0]])
        if unit_lab == "motorized":
            unit_img2 = place_symbol(unit_img2, unit_img, 0, int(
                unit_img2.shape[1]/2-unit_img.shape[1]/2))
        else:
            unit_img2 = place_symbol(
                unit_img2, unit_img, 0, int(unit_img2.shape[1]*0.1))
    else:
        scale_factor = None
        if unit_lab == "mortar":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.3), int(template.shape[0]*0.8)])
            scale_factor = 0.1
        elif unit_lab == "artillery":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.2), int(template.shape[0]*0.2)])
            scale_factor = 0.4
        elif unit_lab == "engineers" or unit_lab == "combat_service":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.6), int(template.shape[0]*0.3)])
            scale_factor = 0.35
        elif unit_lab == "armour":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.9), int(template.shape[0]*0.55)])
            scale_factor = 0.2
        elif unit_lab == "sniper":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.3), int(template.shape[0]*0.2)])
            scale_factor = 0.15
        elif unit_lab == "missile":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.3), int(template.shape[0]*0.6)])
            scale_factor = 0.05
        elif unit_lab == "gun_system":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.3), int(template.shape[0]*0.45)])
            scale_factor = 0.2

        if scale_factor != None:
            unit_img2 = place_symbol(unit_img2, unit_img, int(
                unit_img2.shape[0]*scale_factor), int(unit_img2.shape[1]/2 - unit_img.shape[1]/2))
        elif unit_lab == "wheeled":
            unit_img = cv2.resize(
                unit_img, [int(template.shape[0]*0.4), int(template.shape[0]*0.15)])
            unit_img2 = place_symbol(unit_img2, unit_img, -unit_img.shape[0]-int(
                unit_img2.shape[0]*0.1), int(unit_img2.shape[1]/2 - unit_img.shape[1]/2))
        elif unit_lab == "air_defence":
            unit_img = cv2.resize(unit_img, [template.shape[1], int(
                unit_img.shape[0]*(template.shape[1]/unit_img.shape[1]))])
            unit_img2[-unit_img.shape[0]:, :] = unit_img

    return unit_img2


def add_unit_size(sample: Dict[str, List[np.ndarray]],
                  template: np.ndarray,
                  unit_size_img: np.ndarray,
                  unit_size_label: str) -> np.ndarray:
    """
    Adds unit size symbol above template image. The extra rows will be added above template for unit size.

    Args:
        sample : The dictionary containing the images. Only used when task group symbol is added.
        template : The image for which the unit size will be added
        unit_size_img : The unit size sub-image which will be added.
        unit_size_label : The label of the unit size.

    Returns:
        template : The template where the unit size symbol is added above.
    """
    unit_size_img2 = np.full(
        (int(template.shape[0]*1.2), template.shape[1]), 255)
    unit_size_img2[-template.shape[0]:, -template.shape[1]:] = template
    # Width is used when we place the task group symbol above unit symbol.
    width = 2
    if unit_size_label in ['company', 'battalion', 'regiment']:
        unit_size_img = cv2.resize(
            unit_size_img, [int(template.shape[0]*0.08), int(template.shape[0]*0.18)])
    else:
        unit_size_img = cv2.resize(
            unit_size_img, [int(template.shape[0]*0.18), int(template.shape[0]*0.18)])
    # For these unit sizes we need to add two images
    if unit_size_label in ['half-platoon', 'battalion', 'division']:
        unit_size_img2 = place_symbol(unit_size_img2, unit_size_img, int(
            template.shape[0]*0.01), int(template.shape[1]/2 - unit_size_img.shape[1]*1.05))
        unit_size_img2 = place_symbol(unit_size_img2, unit_size_img, int(
            template.shape[0]*0.01), int(template.shape[1]/2 - unit_size_img.shape[1]*0.05))
        width = 2.5
    # For these unit sizes we need to add three images
    elif unit_size_label in ['platoon', 'regiment']:
        unit_size_img2 = place_symbol(unit_size_img2, unit_size_img, int(
            template.shape[0]*0.01), int(template.shape[1]/2 - unit_size_img.shape[1]/2))
        unit_size_img2 = place_symbol(unit_size_img2, unit_size_img, int(
            template.shape[0]*0.01), int(template.shape[1]/2 - unit_size_img.shape[1]*1.65))
        unit_size_img2 = place_symbol(unit_size_img2, unit_size_img, int(
            template.shape[0]*0.01), int(template.shape[1]/2 + unit_size_img.shape[1]/2*1.3))
        width = 3
    else:  # For these unit sizes we need to add one image
        unit_size_img2 = place_symbol(unit_size_img2, unit_size_img, int(
            template.shape[0]*0.01), int(template.shape[1]/2 - unit_size_img.shape[1]/2))

    # Place the task group symbol aboce unit size symbol
    unit_tactical = False
    if random.uniform(0, 1) > 0.9 and (unit_size_label in ['company', 'battalion', 'regiment']):
        unit_tactical = True
        unit_size_img22 = np.full(
            (int(unit_size_img2.shape[0]*1.05), unit_size_img2.shape[1]), 255)
        unit_size_img22[-unit_size_img2.shape[0]:, -
                        unit_size_img2.shape[1]:] = unit_size_img2
        unit_size_img = get_random('unit_tactical', sample)
        unit_size_img = cv2.resize(
            unit_size_img, [int(template.shape[0]*0.2*width), int(template.shape[0]*0.2)])
        unit_size_img22 = place_symbol(unit_size_img22, unit_size_img, int(
            unit_size_img2.shape[0]*0.01), int(unit_size_img22.shape[1]/2 - unit_size_img.shape[1]/2))
        unit_size_img2 = unit_size_img22
    return unit_size_img2, unit_tactical


def generate_unit(sample: Dict[str, List[np.ndarray]],
                  lab: str,
                  manuever_units: Optional[List[str]] = ['infantry',  # Currently used as global variables
                                                         'anti_tank',
                                                         'armour',],
                  support_units: Optional[List[str]] = ['recce',
                                                        'medic',
                                                        'signal',
                                                        'hq_unit',
                                                        'supply',
                                                        'artillery',
                                                        'mortar',
                                                        'air_defence',
                                                        'sniper',
                                                        'gun_system'],
                  resizable: Optional[List[str]] = ['infantry',
                                                    'anti_tank',
                                                    'recce',
                                                    'medic',
                                                    'signal'],
                  resizable_horizontal: Optional[List[str]] = [
                      'hq_unit', 'supply'],
                  resizable_vertical: Optional[List[str]] = [
                      'motorized', 'cannon'],
                  unit_sizes: Optional[List[str]] = ['team', 'squad', 'half-platoon', 'platoon', 'company',  # Currently repetition because unit sizes are sampled with uniform distirubution
                                                     'team', 'squad', 'half-platoon', 'platoon', 'company',
                                                     'squad', 'half-platoon', 'platoon', 'company',
                                                     'battalion'],  # 'brigade', 'regiment', 'division'
                  can_be_hq: Optional[bool] = False) -> Tuple[np.ndarray, str]:

    composed_label = 'unitsymbol'
    template = get_random('template', sample)

    # Get the random label for image if one is not provided
    if lab == "maneuver":
        unit_lab = manuever_units[randint(0, len(manuever_units)-1)]
        unit_img = get_random(unit_lab, sample)
    elif lab == "support":
        unit_lab = support_units[randint(0, len(support_units)-1)]
    else:
        unit_lab = lab

    # For these two we use line_horizontal from sample
    if 'hq_unit' == unit_lab or 'supply' == unit_lab:
        unit_img = get_random('line_horizontal', sample)
    else:
        unit_img = get_random(unit_lab, sample)

    composed_label += f'__{unit_lab}'

    # Returns unit_img with same shape as template and with symbol in appropriate space.
    unit_img = resize_image(template, unit_img, unit_lab,
                            resizable, resizable_horizontal, resizable_vertical)
    # Add symbol to template.
    template[unit_img == 0] = 0

    # Add additional information to symbol randomly.
    is_motorized = False
    if unit_lab != "armour":
        if random.uniform(0, 1) > 0.8 and unit_lab != 'air_defence':
            unit_img = resize_image(template, get_random(
                'armour', sample), 'armour', resizable, resizable_horizontal, resizable_vertical)
            template[unit_img == 0] = 0
            composed_label += '__armour'
        elif random.uniform(0, 1) > 0.9:
            unit_img = resize_image(template, get_random(
                'line_vertical', sample), "motorized", resizable, resizable_horizontal, resizable_vertical)
            template[unit_img == 0] = 0
            is_motorized = True
            composed_label += '__motorized'
    if random.uniform(0, 1) > 0.9 and not is_motorized:
        unit_img = resize_image(template, get_random(
            'wheeled', sample), "wheeled", resizable, resizable_horizontal, resizable_vertical)
        template[unit_img == 0] = 0
        composed_label += '__wheeled'
    if random.uniform(0, 1) > 0.9:
        unit_img = resize_image(template, get_random(
            'line_vertical', sample), "cannon", resizable, resizable_horizontal, resizable_vertical)
        template[unit_img == 0] = 0
        composed_label += '__cannon'
    if random.uniform(0, 1) > 0.97 and can_be_hq:
        unit_img2 = np.full(
            (int(template.shape[0]*1.6), template.shape[1]), 255)
        unit_img2[:template.shape[0], :template.shape[1]] = template
        unit_img = get_random('line_vertical', sample)
        unit_img = cv2.resize(unit_img, [int(
            unit_img.shape[1]*(int(template.shape[0])/unit_img.shape[0])), int(template.shape[0]*0.6)])
        unit_img2[template.shape[0]:, :unit_img.shape[1]][unit_img == 0] = 0
        template = unit_img2
        composed_label += '__hq_vertical'

    # Add unit size
    # Generate random unit size
    unit_size_lab = unit_sizes[randint(0, len(unit_sizes)-1)]
    image_lab = unit_size_lab
    if unit_size_lab in ['half-platoon', 'platoon']:
        image_lab = "squad"
    elif unit_size_lab in ['battalion', 'regiment']:
        image_lab = "company"
    elif unit_size_lab in ['division', 'regiment']:
        image_lab = "brigade"

    composed_label += f'__{unit_size_lab}'

    unit_size_img = get_random('size_' + image_lab, sample)

    template, unit_tactical = add_unit_size(
        sample, template, unit_size_img, unit_size_lab)

    if unit_tactical:
        composed_label += '__unit_tactical'
    return template.astype('float32'), composed_label
