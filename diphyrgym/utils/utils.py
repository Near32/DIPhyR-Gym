'''
Licence.
'''
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


def update_plot(angles, velocities, fig, ax=None, line=None):
    '''
    Function to update the plot
    '''
    if line is None:
        assert ax is not None
        line, = ax.plot(angles, velocities, 'g-')
    else:
        line.set_data(angles, velocities)
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return img

def unwrap_angles(angles):
    '''
    Unwrap an angle in radians to be continuous.
    '''
    return np.unwrap(angles)


def STR2BT(sentences, max_sentence_length=0):
    '''
    string to byte tensor.
    '''
    if isinstance(sentences, str):
        sentences = [sentences]
    btss = []
    for s in sentences:
        bts = np.asarray(list(bytes(s, 'utf-8')), dtype=np.uint8)
        if max_sentence_length < bts.shape[0]:  max_sentence_length = bts.shape[0]
        btss.append(bts)
    ret = np.zeros((len(btss), max_sentence_length), dtype=np.uint8)
    for bts_idx, bts, in enumerate(btss):
        ret[bts_idx, :bts.shape[0]] = bts
    return ret

def BT2STR(bt):
    '''
    Byte tensor to string.
    '''
    sentences = []
    for idx in range(bt.shape[0]):
        sentence = "".join(map(chr,bt[idx].tolist())).replace('\x00','')
        sentences.append(sentence)
    return sentences


def randomize_MJCF(
    base_filepath,
    output_filepath,
    rgbdName2tagName2attrName={
       "pole": {
         "geom": ["fromto"],
         "velocity/linear": ["x", "y", "z"],
         "velocity/angular": ["x", "y", "z"],
       }, 
       "cart": {
         "geom": ["size"],
         "velocity/linear": ["x", "y", "z"],
         "velocity/angular": ["x", "y", "z"],
       }, 
    },
    np_random=np.random,
):
    '''
    Function to randomize MuJoCo XML parameters
    As it assumes the inertiafromgeom=True attribute in compiler tag,
    this function should randomise some geom tags in order to randomise inertia parameters.
    The range of acceptable values should be specified with relevant attributes of the randomised
    tag, e.g.: maxsize for geom with size attribute, or maxx for linear/angular velocity tags.
    '''
    tree = ET.parse(base_filepath)
    root = tree.getroot()
    
    for rgbdName, tagName2attrName in rgbdName2tagName2attrName.items():
        for body in root.findall('.//body'):
            if body.get('name') == rgbdName:
                for tagBranch, attrName in tagName2attrName.items():
                    tagBranch = tagBranch.split("/")
                    # Retrieve child tag from branch:
                    node = body
                    for tag in tagBranch:
                        node = node.find(tag)
                    if node is None:
                        continue
                    tag = node
                    for attr in attrName:
                        if attr in tag.attrib:
                            minv = tag.attrib[f"{attr}min"].split(" ")
                            maxv = tag.attrib[f"{attr}max"].split(" ")
                            attr_value = " ".join([
                              str(np_random.uniform(float(minv[vidx]), float(maxv[vidx])))
                              for vidx in range(len(minv))
                            ])
                            tag.set(attr, attr_value) 
    
    # Save the modified XML file
    tree.write(output_filepath)
    return output_filepath

