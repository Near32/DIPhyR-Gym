import numpy as np
import xml.etree.ElementTree as ET

# Function to randomize MuJoCo XML parameters
# As it assumes the inertiafromgeom=True attribute in compiler tag,
# this function should randomise some geom tags in order to randomise inertia parameters.
# The range of acceptable values should be specified with relevant attributes of the randomised
# tag, e.g.: maxsize for geom with size attribute, or maxx for linear/angular velocity tags.
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

