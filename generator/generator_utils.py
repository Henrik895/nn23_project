import numpy as np
import cv2
import re
import os
from math import sin, cos, tan, radians, ceil
import random
from random import randint
from scipy import ndimage
from generate_unit_symbol import *
import copy
from typing import Tuple, List, Dict, Optional

def place_symbol(canvas : np.ndarray,
                 symbol : np.ndarray,
                 point1 : int,
                 point2 : int) -> np.ndarray:
    """
    Places the given symbol on canvas. Only places the black pixels.

    Args:
        canvas : The overall image which is generated
        symbol : The image which will be placed on canvas
        point1 : The upper edge on canvas where the symbol will be placed.
        point2 : The left edge on canvas where the symbol will be placed.

    Returns:
        canvas: A image with placed symbol.
    """

    canvas[point1:point1+symbol.shape[0],point2:point2+symbol.shape[1]][symbol < 140] = 0
    return canvas

def get_points(dim : Tuple[int,int],
               symbol : np.ndarray,
               locations : List[Tuple[Tuple[int,int],Tuple[int,int]]],
               locations_units : List[Tuple[Tuple[int,int],Tuple[int,int]]],
               location_placement : List[Tuple[Tuple[int,int],Tuple[int,int]]]) -> Tuple[int,int]:
    """
    Finds the empty place randomly on canvas where to put the symbols.

    Args:
        dim : The dimensions of canvas.
        symbol : The image for which the location will be looked for.
        locations : The locations of another symbols.
        locations_units : The locations of unit symbols.
        locations_placment : The locations of placement symbols.

    Returns:
        Upper-left point where the symbol can be placed.

    """
    is_overlap = True
    counter = 0
    while(is_overlap):
        overlap = False
        #Get the random location on canvas
        point1 = randint(0,dim[0]-symbol.shape[0]-10)
        point2 = randint(0,dim[1]-symbol.shape[1]-10)
        overlap = (check_overlap(point1,point2,locations,symbol.shape) or
                  check_overlap(point1,point2,locations_units,symbol.shape) or
                  check_overlap(point1,point2,location_placement,symbol.shape))

        if not overlap:
            is_overlap = False

        counter+=1
        if counter > 50:
            raise Exception("No place found")
        
    
    return point1, point2

def get_grid_numbers(grid : int,
                     scale : float,
                     sample_extras : Dict[str,List[np.ndarray]]) -> Tuple[np.ndarray,np.ndarray]:
    """
    Samples the grid number images.

    Args:
        grid : The MGRS gridnumber
        scale : The scale for the number images
        sample_extras : The dictionary which will contain the numbers images.

    Returns:
        The images of numbers. If the number is 9, then 0 and 9 will be returned.

    """
    first_number = get_random(str(grid//10), sample_extras)
    first_number = resize_by_scale(first_number, scale)
    second_number = get_random(str(grid%10), sample_extras)
    second_number = resize_by_scale(second_number, scale)
    return first_number, second_number


def add_grid_number(canvas : np.ndarray,
                    point1 : int,
                    offset0 : int,
                    offset1 : int,
                    grid : int,
                    placement : np.ndarray,
                    sample_extras : Dict[str,np.ndarray],
                    scale : Optional[float] = 0.5) -> np.ndarray:
    """
    Adds the grid numbers for placement symbol onto the canvas. Adds on left of the symbol.

    Args:
        canvas : The overall image which is generated. The numbers will be palced onto that.
        point1 : The height on placment symbol where the number will be added
        offset0 : The offset along the y-axis (height).
        offset1 : The offset along the x-axis (width).
        grid : The MGRS grid number
        placement :  The placement symbol for which the numbers will be added.
        sample_extras : The dictionary which will contain the numbers images.
        scale : The scaling factor for the grid number.
    
    Return:
        Canvas image with placed numbers.
    """
    first_number, second_number = get_grid_numbers(grid, scale, sample_extras)
    point1_1 = offset0+point1-int(first_number.shape[0]/2)
    point2_1 = canvas.shape[1]-offset1-placement.shape[1]-10
    canvas = place_symbol(canvas, first_number, point1_1, point2_1-second_number.shape[1]-first_number.shape[1]-int(15*scale))
    canvas = place_symbol(canvas, second_number, point1_1, point2_1-second_number.shape[1]-5)
    return canvas

def add_grid_number2(canvas : np.ndarray,
                    point2 : int,
                    offset0 : int,
                    offset1 : int,
                    grid : int,
                    placement : np.ndarray,
                    sample_extras : Dict[str,List[np.ndarray]],
                    scale : Optional[float] = 0.5) -> np.ndarray:
    """
    Adds the grid numbers for placement symbol onto the canvas. Adds on bottom of the symbol.

    Args:
        canvas : The overall image which is generated. The numbers will be palced onto that.
        point2 : The width on placment symbol where the number will be added
        offset0 : The offset along the y-axis (height).
        offset1 : The offset along the x-axis (width).
        grid : The MGRS grid number
        placement :  The placement symbol for which the numbers will be added.
        sample_extras : The dictionary which will contain the numbers images.
        scale : The scaling factor for the grid number.
    
    Return:
        Canvas image with placed numbers.
    """
    first_number, second_number = get_grid_numbers(grid, scale, sample_extras)
    point1_1 = offset0+placement.shape[0]+10
    point2_1 = canvas.shape[1]-offset1-point2
    canvas = place_symbol(canvas, first_number, point1_1, point2_1-first_number.shape[1]-10)
    canvas = place_symbol(canvas, second_number, point1_1, point2_1+10)
    return canvas

def draw_line(canvas : np.ndarray,
              point1 : int,
              point2 : int,
              rotation : int,
              side : int,
              img : np.ndarray) -> np.ndarray:
    """
    Drawns onse-side of the line trough the given symbol (img).

    Args:
        canvas : The overall image which is generated. The numbers will be palced onto that.
        point1 : The img upper edfe locations on canvas
        point1 : The img left edge locations on canvas
        rotation : The rotation of the image.
        side : Values can be (1 & -1). Indicator for which side (left or right) of the line will be drawn.
        img : The symbol.
    
    Return:
        Canvas image with drawn line.
    """

    # Find the center of the symbol. Will be the center of line
    center = [int(point1+img.shape[0]/2), int(point2+img.shape[1]/2)]
    # Keep track of drawing to make sure we don't go out of bounds
    max_lengths = np.zeros((2,2),dtype="i")
    max_lengths[0,1] = -canvas.shape[0]+center[0] #Up
    max_lengths[0,0] = -canvas.shape[1]+center[1] #Right
    max_lengths[1,1] = center[0] #Down
    max_lengths[1,0] = center[1] #Left
    prev_point = center
    rotation = rotation + side*90
    for i in range(0,randint(3,6)):
        #Random lenght of the line
        line_length = randint(70,150)
        # Get the next points location
        point2_1, point1_1 = point_location(rotation,(line_length,line_length),prev_point,(1,1))
        new_point = np.array((point1_1,point2_1))

        max_lengths[0,:] = max_lengths[0,:] + (prev_point-new_point)
        max_lengths[1,:] = max_lengths[1,:] + (prev_point-new_point)
        if (max_lengths[0,0] > 0 and max_lengths[0,1] > 0 and max_lengths[1,0] < 0 and max_lengths[1,1] < 0):
            break
        canvas = canvas.astype(np.uint8).copy()
        # For cv2 draw the x-axis (width) is first.
        cv2.line(canvas, (prev_point[1],prev_point[0]), (new_point[1],new_point[0]), 0, 9)
        prev_point = new_point
        rotation = rotation+randint(-20,20)
    return canvas

def point_location(rotation : int,
                   symbol_shape : Tuple[int,int],
                   center : Tuple[int,int],
                   unit_symbol_shape : Tuple[int,int]) -> Tuple[int,int]:
    """
    Given the rotation finds the upper left point on canvas where to place the unit symbol for given tactical task.

    Args:
        rotation : The symbols (tactical task) rotation
        symbol_shape : The dimensions of symbol image
        center : The center of the symbol
        unit_symbol_shape : The shape of the unit symbol
    
    Return:
        The upper left point for the unit_symbol
    """
    x = center[1]
    y = center[0]
    
    if rotation < 180:
        y -= int((rotation/180)*unit_symbol_shape[0])
        x -= int(unit_symbol_shape[1]/2 * abs(cos(radians(rotation))))
    else:
        y -= int(((360-rotation)/180)*symbol_shape[0])
        x -= int(unit_symbol_shape[1]/2 * abs(cos(radians(rotation))) + unit_symbol_shape[1]/2 * abs(sin(radians(rotation))))
    
    #Point on ellipse line
    theta = tan(radians(rotation))**2
    if theta == 0:
        theta = 1e-10
    y2 = int(symbol_shape[0]*symbol_shape[1] / (np.sqrt(symbol_shape[0]**2 + symbol_shape[1]**2 * theta)))
    x2 = int(symbol_shape[0]*symbol_shape[1] / (np.sqrt(symbol_shape[1]**2 + symbol_shape[0]**2 / theta)))
    if rotation >= 90 and rotation < 270:
        y -= y2
    else:
        y += y2
    if (rotation >= 0 and rotation < 180):
        x += x2
    else:
        x -= x2
    return x,y

def check_overlap(point1 : int,
                  point2 : int,
                  locations : List[Tuple[Tuple[int,int],Tuple[int,int]]],
                  symbol_shape : Tuple[int,int],
                  max_overlap : Optional[int] = 50) -> bool:
    """
    Given a symbol and the locations, check if the symbol overlaps with any.

    Args:
        point1 : The upper edge locations for symbol
        point2 : The left edge locations for the symbol
        locations : The locations of the current symbols on canvas
        symbol_shape : The shape of the symbol
        max_overlap : The maximum overlap in pixels which is allowed along the axis.
    
    Return:
        The boolean value whether there is overlap
    """
    is_overlap = False
    for location in locations:
        if ((point1 < location[1][0]-max_overlap) and
            (point2 < location[1][1]-max_overlap) and
            (point1+symbol_shape[0] > location[0][0]+max_overlap) and
            (point2+symbol_shape[1] > location[0][1]+max_overlap)):
            is_overlap = True
            break
    return is_overlap

def read_into_dic(directory : str,
                  re_in : str,
                  output_dir : Optional[Dict[str,List[np.ndarray]]] = None,
                  excess_str : Optional[int] = 100) -> Dict[str,List[np.ndarray]]:
    """
    Reads all images in a directory into a dictionary and returns them as the dictionary of numpy arrays where the labels are keys.

    Args:
        directory : The path to the directory containing the images to read.
        re_in : The regular expression pattern to match the label in the filename of each image.
        output_dir : The output dictionary where the images will be stored. If none is given, then the new dictioanry will be greated.
        excess_str : The threshold value to cut excess edges on the images.

    Returns:
        A dictionary containing of numpy arrays representing the images, and the keys are labels corresponding to images in the array.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Check if the directory is given. If given then add to the existing directory,
    if output_dir == None:
        output_dir = {}
    for filename in os.listdir(directory+"/"):
        # Check if file is an image
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.threshold(img, excess_str, 255, cv2.THRESH_BINARY)[1]
        img = cut_excess_white(img)
        key = re.findall(re_in, filename)[0]

        if key in output_dir:
            output_dir[key].append(img)
        else:
            output_dir[key] = [img]
    return output_dir

def read_into_list(directory : str,
                   re_in : str,
                   excess_str : Optional[int] = 100) -> Tuple[List[np.ndarray],List[str]]:
    """
    Reads all images in a directory and returns them as a list of numpy arrays, along with their corresponding labels list.

    Args:
        directory : The path to the directory containing the images to read.
        re_in : The regular expression pattern to match the label in the filename of each image.
        excess_str : The threshold value.

    Returns:
        A tuple containing a list of numpy arrays representing the images, and a list of labels corresponding to each image.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    output_list = []
    output_labels = []
    for filename in os.listdir(directory+'/'):
        # Check if file is an image
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.threshold(img, excess_str, 255, cv2.THRESH_BINARY)[1]
        img = cut_excess_white(img)
        output_list.append(img)
        output_labels.append(re.findall(re_in, filename)[0])

    return output_list, output_labels

def resize_by_scale(img : np.ndarray,
                    scale : float) -> np.ndarray:
    """
    Resize the image by scale

    Args:
        img : The image to be scaled.
        scale : The scale factor.
    
    Returns:
        The scaled iamge
    
    Raises:
        TypeError: If the image is not numpy.ndarray
        TypeError: If the scale is not a number.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if not isinstance(scale, float):
        raise TypeError("scale must be a float")

    img = cv2.resize(img, (int(img.shape[1]*scale),int(img.shape[0]*scale)))
    return img

def cut_excess_white(img : np.ndarray,
                     excess_str : Optional[int] = 120) -> np.ndarray:
    """
    Removes excess rows and columns from a image edges.

    Args:
        img : The image for which the excess edges will be removed.
        excess_str : The upper value of grayscale pixel for which is not consider a background.
    
    Returns:
        The image without excessive columns and rows on the edges.
    
    Raises:
        TypeError: If the image is not numpy.ndarray.
        ValueError: When the excess_str is not between 0 and 255.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if not 0 <= excess_str <= 255:
        raise ValueError("excess_str must be between 0 and 255")
    
    img = img[np.argwhere(np.amin(img,axis=1) < excess_str)[0][0]:np.argwhere(np.amin(img,axis=1) < excess_str)[-1][0],:]
    img = img[:,np.argwhere(np.amin(img,axis=0) < excess_str)[0][0]:np.argwhere(np.amin(img,axis=0) < excess_str)[-1][0]]
    return img

def read_in_labels(file : str) -> Dict[str,int]:
    """
    Reads in the labels file and assigns the numeric value for each label. The numeric value is the labels row in the file.

    Args:
        file : The file which contains the labels as string

    Returns:
        The dictionary where the key is label and value if integer.
    
    TODO file type checking 
    """
    labels_to_nr = {}
    i = 0
    with open(file) as f:
        for line in f:
            labels_to_nr[line.strip('\n')] = i
            i += 1
    return labels_to_nr

def get_labels(label : str,
               labels_to_nr : Dict[str,int]) -> int:
    """
    Return numerical value of the label

    Args:
        label : The label for which the numerical value is given
        labels_to_nr : The dictionary which maps the string to integers

    Return:
        Label as integer
    
    Raises:
        TypeError: The label is not string
    """

    if not isinstance(label, str):
        raise TypeError("label must be str")

    return labels_to_nr[label]

def get_locations(data_locations : List[Tuple[Tuple[int,int],Tuple[int,int]]],
                  image_height : int, image_width : int,
                  offset : Optional[int] = 0) -> np.ndarray:
    """
    Computes normalized locations and dimensions of objects in an image.

    Args:
        data_locations : A list of tuples where each tuple contains two points (top left and bottom right) that define
                              the location of an object in the image.
        image_height : The height of the image (canvas).
        image_width : The width of the image (canvas).
        offset : The offset to add to the y coordinate of the bounding box. Used when the image is padded to square

    Returns:
        An array where each row corresponds to an object and contains 4 elements:
        - The y-coordinate of the center of the object (normalized to [0,1]).
        - The x-coordinate of the center of the object (normalized to [0,1]).
        - The height of the object (normalized to [0,1]).
        - The width of the object (normalized to [0,1]).
    """

    data_locations = np.array(data_locations)
    data_locations2 = np.zeros((data_locations.shape[0],4))
    data_locations2[:,2] = (data_locations[:,1,1] - data_locations[:,0,1])/image_height
    data_locations2[:,3] = (data_locations[:,1,0] - data_locations[:,0,0])/image_width
    data_locations2[:,0] = (data_locations[:,1,1] + data_locations[:,0,1])/(2*image_height)
    data_locations2[:,1] = ((data_locations[:,1,0] + data_locations[:,0,0])/2 + offset) / image_width
    return data_locations2

def find_cap(img : np.ndarray,
             excess_str : Optional[int] = 120) -> Tuple[int,int]:
    """
    Find the location of the cap between letters C/G/S in symbol

    Args:
        img (numpy.ndarray): A grayscale image represented as a 2D numpy array.
        excess_str (int): The threshold value for what is considered a "white" pixel.

    Returns:
        Tuple of left and right bounds of the capital letter in the image.
    """

    values = np.argwhere(np.amin(img,axis=0) > excess_str)

    val_prev = values[0][0]
    sizes = []
    cuts = []
    cuts.append(0)
    size = 1
    for i in range(1,len(values)):
        val = values[i][0]
        if val-val_prev == 1:
            val_prev = val
            size += 1
        else:
            val_prev = val
            sizes.append(size)
            cuts.append(i)
            size = 1
    sizes.append(size)
    cuts.append(i)

    loc = np.argmax(sizes) + 1
    left = values[cuts[loc-1]][0]
    right = values[cuts[loc]-1][0]
    return left, right

def rotate_img(img : np.ndarray,
               rotation : int,
               padding_value : Optional[int] = 255) -> np.ndarray:
    """
    Rotate an image by a given angle.

    Parameters:
        img : The image to be rotated.
        rotation : The rotation angle in degrees.
        padding_val : The value to use for padding.

    Returns:
        The rotated image.
    """
    
    img_rotated = ndimage.rotate(img, rotation, mode='constant',cval=padding_value)
    return img_rotated

def add_unit_symbol_in_middle(img : np.ndarray,
                              scale : float,
                              sample_units : Dict[str,List[np.ndarray]],
                              manuever_units : List[str],
                              support_units : List[str],
                              resizable : List[str],
                              resizable_horizontal : List[str],
                              resizable_vertical : List[str],
                              unit_sizes : List[str]) -> Tuple[np.ndarray, str, int, Tuple[int,int], Tuple[int,int]]:
    """
    Add a unit symbol in the middle of the image.

    Parameters:
    image : The input image (cover/guard/screen symbol).
    scale (float): The scaling factor.
    sample_units (dir): The dictionary containing unit symbol samples.
    maneuver_units (list): A list of maneuver units.
    support_units (list): A list of support units.
    resizable (list): A lsit of resizable symbols.
    resizable_horizontal (list): A lsit of resizable horizontal symbols.
    resizable_vertical (list): A list of resizable vertical symbols.
    unit_sizes (list): A list of unit sizes from which to sample.

    Returns:
        A tuple containing the following elements:
        - The output image with the unit symbol in the middle.
        - The label of the unit.
        - The rotation angle in degrees.
        - The top-left point of the image without symbol.
        - The bottom-right point of the image without symbol.
    """
    
    #Cut excess white from edges
    img = cut_excess_white(img)

    img, rotation = augment(img, apply_rotation=False, apply_transformation=True, apply_boldness=True, scale_to_binary=True)

    # Find the cap in cover/guard/screen symbol
    left, right = find_cap(img)

    #Get unit_symbol  
    unit_symbol, unit_lab = generate_unit(sample_units,"maneuver",
                                manuever_units,support_units,
                                resizable,resizable_horizontal,
                                resizable_vertical,unit_sizes,False)

    unit_symbol = resize_by_scale(unit_symbol, scale*0.7)

    if img.shape[0] < ceil(unit_symbol.shape[0]/2):
        shape1 = unit_symbol.shape[0]
    else:
        shape1 = ceil(unit_symbol.shape[0]/2) + img.shape[0]

    img2 = np.full([int(shape1),left+unit_symbol.shape[1]+(img.shape[1]-right) + 14], 255) #+14 is for the gap between arrows and unit symbol
    
    # Place the unit symbol into middle
    img2 = place_symbol(img2, unit_symbol, 0, left+7)
    # We will also do the same operations for additional image with a dummy symbol.
    # This is needed to get the cover/guard/screen locations without unit_symbol
    img3 = copy.deepcopy(img2)
    # Place the left side of the screen/cover/guard
    img2 = place_symbol(img2, img[:,0:left], ceil(unit_symbol.shape[0]/2),0)
    # Place the right side of the screen/cover/guard
    img2 = place_symbol(img2, img[:,right:], ceil(unit_symbol.shape[0]/2), left+unit_symbol.shape[1]+14)
    #img2 = img2.astype('float32')
    
    img3[img3 < 100] = 150
    img3 = place_symbol(img3, img[:,0:left], ceil(unit_symbol.shape[0]/2),0)
    img3 = place_symbol(img3, img[:,right:], ceil(unit_symbol.shape[0]/2), left+unit_symbol.shape[1]+14)
    #img3 = img3.astype('float32')
    
    img2pad = np.pad(img2, ((600,600),(600,600)), "constant", constant_values=255)
    img3pad = np.pad(img3, ((600,600),(600,600)), "constant", constant_values=255)
    
    rotation = randint(0,359)
    img2_rotated = ndimage.rotate(img2pad, rotation, reshape=False, mode='constant',cval=255)
    img3_rotated = ndimage.rotate(img3pad, rotation, reshape=False, mode='constant',cval=255)
    
    top1 = np.argwhere(np.amin(img3_rotated,axis=1) < 110)[0][0]
    bottom1 = np.argwhere(np.amin(img3_rotated,axis=1) < 110)[-1][0]
    left1 = np.argwhere(np.amin(img3_rotated,axis=0) < 110)[0][0]
    right1 = np.argwhere(np.amin(img3_rotated,axis=0) < 110)[-1][0]
    
    top2 = np.argwhere(np.amin(img2_rotated,axis=1) < 110)[0][0]
    bottom2 = np.argwhere(np.amin(img2_rotated,axis=1) < 110)[-1][0]
    left2 = np.argwhere(np.amin(img2_rotated,axis=0) < 110)[0][0]
    right2 = np.argwhere(np.amin(img2_rotated,axis=0) < 110)[-1][0]
    
    point1 = (top1-top2,left1-left2)
    point2 = (bottom1-bottom2,right1-right2)
    
    img2_rotated = cut_excess_white(img2_rotated)
    #img3_rotated = cut_excess_white(img3_rotated)

    img2_float = img2_rotated.astype('float32') #OpenCV requires float32 type, cant work with int16
    
    return img2_float, unit_lab, rotation, point1, point2

"""
def get_points_after_rotation(point : Tuple[int,int],
                              rotation : int) -> Tuple[int,int]:
    x = point[1]*cos(radians(rotation))-point[0]*sin(radians(rotation))
    y = point[1]*sin(radians(rotation))+point[0]*cos(radians(rotation))
    return (y,x)
"""

def get_mortar_area_img(number : int,
                        scale : float,
                        sample_extras : Dict[str,List[np.ndarray]]) -> np.ndarray:
    """
    Samples a mortar area image given the number.

    Args:
        number : The number for mortar area.
        scale : The scale factor.
        sample_extras : The dictionary containg the images.
    
    Return:
        The mortar area image.
    """
    # Sample a random circle
    mortar_img = get_random('mortar', sample_extras)
    # Sample a random letter m
    m_letter = get_random('m', sample_extras)
    m_letter = cv2.resize(m_letter, (int(m_letter.shape[1]*0.5),int(m_letter.shape[0]*0.5)))
    # Sample a random number image
    number = get_random(str(number), sample_extras)
    number = cv2.resize(number, (int(number.shape[1]*0.5),int(number.shape[0]*0.5)))
    # Get a center of circle
    center = (int(mortar_img.shape[0]/2),int(mortar_img.shape[1]/2))
    # Place the m and number
    mortar_img = place_symbol(mortar_img, m_letter, center[0]-int(m_letter.shape[0]/2),center[1]-10-m_letter.shape[1])
    mortar_img = place_symbol(mortar_img, number, center[0]-int(number.shape[0]/2),center[1]+10)
    mortar_img = cv2.resize(mortar_img, (int(mortar_img.shape[1]*scale),int(mortar_img.shape[0]*scale)))
    return mortar_img

def get_random(label : str,
               sample : Dict[str,List[np.ndarray]]) -> np.ndarray:
    """
    Chooses random image from the sample list given the label.

    Args:
        label : The specific images from which we wish to sample.
        sample : The dictionary containing the images.

    Return:
        Sampled image which corresponds to label.
    """

    imgs = sample[label]
    return np.copy(imgs[randint(0,len(imgs)-1)])

def get_noise_img(sample_extras : Dict[str,List[np.ndarray]]) -> np.ndarray:
    """
    Like get_random but with predefined scale for the image. Used to keep the main code shorter.

    Args:
        sample : The dictionary containing the images.

    Return:
        Sampled image of noise.
    """
    noise_img = get_random('noise', sample_extras)
    noise_img = resize_by_scale(noise_img, 0.17)
    return noise_img

def get_exercise_text(scale : float,
                      sample : Dict[str,List[np.ndarray]],
                      language : str) -> np.ndarray:
    """
    Generates an image of the word 'exercise' in English or Estonian.

    Args:
        scale : The scale factor to resize the images.
        sample : A dictionary which contains letters images.
        language : The language of the text. Either 'en' for English or 'et' for Estonian.

    Returns:
        np.ndarray: The image of the text.
    """

    letters_dic = {}
    max_height = 0
    length = 0
    if language == 'en':
        letters = ['e','x','e','r','c','i','s','e']
    else:
        letters = ['oline','p','p','u','s']
    for letter in set(letters):
        letter_img = get_random(letter, sample)
        letter_img = resize_by_scale(letter_img, scale)
        if letter_img.shape[0] > max_height:
            max_height = letter_img.shape[0]
        length += letter_img.shape[1]
        letters_dic[letter] = letter_img
    
    max_height += 3

    if language == 'en':
        length += int(7*6) #Add 4pixels between symbols
        length += int(letters_dic['e'].shape[1]*2)
    else:
        length += int(4*6)
        length += letters_dic['p'].shape[1]
    
    text_img = np.full((max_height,length),255)
    pointer = 0

    for letter in letters:
        text_img = place_symbol(text_img, letters_dic[letter],randint(0,3),pointer)
        pointer+=letters_dic[letter].shape[1]+6

    return text_img

def place_exercise_text(canvas : np.ndarray,
                        scale : float,
                        sample: Dict[str,List[np.ndarray]],
                        language : Optional[str] = None) -> np.ndarray:
    """
    Places the text 'exercise' in English or Estonian on an image.

    Args:
        canvas : The input image to place the text on.
        scale : The scale factor to resize the images.
        sample : A dictionary of images.
        language : The language of the text. Either 'en' for English or 'et' for Estonian. Defaults to a random language.

    Returns:
        The image with the placed text.
    """

    if not isinstance(canvas, np.ndarray):
        raise TypeError("img must be a NumPy array")

    if language is None:
        language = ['en', 'et'][randint(0,1)]
    elif not isinstance(language, str):
        raise TypeError("language must be a str")
    
    scale = scale*0.7
    vertical_loc = randint(16,28)
    mid = int(canvas.shape[1]/2)

    for i in range(2):
        ex_texts = [get_exercise_text(scale, sample, language) for _ in range(3)]
        lines = [resize_by_scale(get_random('line', sample), scale) for _ in range(2)]
        lines = [cv2.resize(line, (int(line.shape[1]*0.5), line.shape[0])) for line in lines]

        canvas = place_symbol(canvas, ex_texts[0], vertical_loc, mid-int(ex_texts[1].shape[1]/2)-lines[0].shape[1]-ex_texts[0].shape[1]-8)
        canvas = place_symbol(canvas, lines[0], vertical_loc+int(ex_texts[0].shape[0]/2)-int(lines[0].shape[0]/2), mid-int(ex_texts[1].shape[1]/2)-lines[0].shape[1]-4)
        canvas = place_symbol(canvas, ex_texts[1], vertical_loc, mid-int(ex_texts[1].shape[1]/2))
        canvas = place_symbol(canvas, lines[1], vertical_loc+int(ex_texts[1].shape[0]/2)-int(lines[1].shape[0]/2), mid+int(ex_texts[1].shape[1]/2)+4)
        canvas = place_symbol(canvas, ex_texts[2], vertical_loc, mid+int(ex_texts[1].shape[1]/2)+lines[1].shape[1]+8)

        vertical_loc = canvas.shape[0]-int(1.1*np.maximum(np.maximum(ex_texts[0].shape[0],ex_texts[1].shape[0]),ex_texts[2].shape[0]))

    return canvas

def inverse(img : np.ndarray,
            binary_threshold : Optional[int] = 110) -> np.ndarray:
    """
    Inverse the black and white pixels of an image.

    Parameters:
        img : The image to be rotated.
        binary_threshold : The value to from which the threshold is applied.

    Returns:
        The inverse image.
    """

    # Set the background pixels to 0 and symbols pixels to 255
    img = np.where(img <= binary_threshold, 255, 0)
    return img

def augment(img : np.ndarray,
            remove_excess : Optional[bool] = True,
            excess_str : Optional[int] = 110,
            apply_flip : Optional[bool] = False,
            flip_random : Optional[bool] = True,
            apply_rotation : Optional[bool] = False,
            rotation : Optional[int] = None,
            padding_val : Optional[int] = 255,
            apply_transformation : Optional[bool] = False,
            transformation_dir : Optional[int] = None,
            apply_boldness : Optional[bool] = False,
            boldness_dir : Optional[int] = None,
            boldness_str : Optional[int] = None,
            add_noise : Optional[bool] = False,
            noise_threshold  : Optional[float] = 0.999,
            scale_to_binary : Optional[bool] = False,
            binary_threshold : Optional[int] = 110,
            invert : Optional[bool] = False,
            normalize : Optional[bool] = False) -> Tuple[np.ndarray,int]:
    """
    Augments the given image

    Args:
        img : The image to be augmented.
        remove_excess : A boolean indicating whether to remove excess rows and columns that appeared after rotation and padding.
        excess_str : The upper value of grayscale pixel for which is not consider a background.
        apply_flip : A boolean indicating whether to flip the image along the vertical axis.
        flip_random : A boolean indicating whether to randomly flip the image. 
        apply_rotation : A boolean indicating whether to apply random rotation to the image.
        rotation : An integer indicating the value of rotation angle.
        padding_val : An integer indicating the padding value to be used.
        apply_transformation : A boolean indicating whether to apply affine transformation to the image.
        transformation_dir : An integer indicating the direction of transformation.
        apply_boldness : A boolean indicating whether to apply dilation or erosion to the image.
        boldness_dir : An integer indicating the direction of dilation or erosion.
        boldness_str : An integer indicating the strength of dilation or erosion.
        add_noise : A boolean indicating whether to add random noise to the image.
        noise_threshold  : A float indicating the threshold for adding noise.
        scale_to_binary : A boolean indicating whether to scale the image to binary.
        binary_threshold : An integer indicating the threshold for scaling to binary.
        invert : A boolean indicating whether to invert the image
        normalize :  A boolean indicating whether to normalize the image.
    
    Return:
        img : The augmented image.
        rotation : The rotation angle of the image
    """
      
    #Select random image subclass
    #Randomize the size of the image
    #Flip the image along vertical axis.
    if apply_flip:
        if flip_random:
            if randint(0,1) == 0:
                img = cv2.flip(img,1)
        else:
            img = cv2.flip(img,1)
    #Randomize the rotation of the image.
    if apply_rotation:
        if rotation == None:
            rotation = randint(0,359)
        img = rotate_img(img,rotation)
    else:
        rotation = 0
    if apply_transformation:
        #Add padding for affine transformation. Otherwise the picture might not be in bounds.
        img = np.pad(img, (1000, 1000), 'constant', constant_values=(padding_val, padding_val))
        #Originial right triangle
        pts1 = np.float32([[3,3],[3,10],[10,3]])
        if transformation_dir == None:
            transformation_dir = randint(0,2)
        if transformation_dir == 0:
            pts2 = np.float32([[randint(1,5),randint(1,5)],[3,10],[10,3]])
        if transformation_dir == 1:
            pts2 = np.float32([[3,3],[randint(1,5),randint(8,12)],[10,3]])
        if transformation_dir == 2:
            pts2 = np.float32([[3,3],[3,10],[randint(8,12),randint(1,5)]])
        #Get transformation
        M = cv2.getAffineTransform(pts1,pts2)
        #Apply transformation
        img  = cv2.warpAffine(img ,M,(img.shape[1],img.shape[0]),borderValue = 255)
    #Remove excess rows and columns that appeared after rotation and padding
    if remove_excess:
        img = cut_excess_white(img, excess_str)
    
    #Currently with random we do not dialte(that means we do not erode, some symbols disapper with that)
    #dilation and erosion
    if apply_boldness:
    #If randint = 2, then we apply neither.
        kernel = np.ones((3,3),np.uint8) #Kernel 3x3 seemed to work fine.
        if boldness_dir == None:
            boldness_dir = randint(0,1) #If 0 we apply neither.
        if boldness_str == None:
            boldness_str = randint(1,5)
        if boldness_dir == 1: #erode 1 iteration. #OpenCV erodes white to black. In our case function erode actually dilates.
            img = cv2.erode(img,kernel,iterations = boldness_str)
        if boldness_dir == -1:
            img = cv2.dilate(img,kernel,iterations = boldness_str)
    
    #Apply noise by changing random pixels to one.
    if add_noise:
        img[np.random.rand(img.shape[0],img.shape[1]) > noise_threshold] = 0
    
    #Changes all the pixels with drawing to one and all the "empty" pixels to 0.
    if scale_to_binary:
        if invert:
            img = inverse(img)
        else:
            img[img <= binary_threshold] = 0
            img[img > binary_threshold] = 255
            
    if normalize:
        img = img / 255
    
    return img, rotation
