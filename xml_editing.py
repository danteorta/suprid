import xml.etree.ElementTree as etree
import pandas as pn
from io import StringIO
import numpy as np

# Open the XML file
files_path = 'D:\\Users\dorta\Dropbox\Stanford\Research\work' \
            'space\\test_run\DiscretizationToolkit\\'
orig_filename = 'xDante.xml'
tree = etree.parse(files_path + orig_filename)
root = tree.getroot()

# Read the relevant features
# mesh = root[0]
data = root.find('Mesh').find('Data')
aperture = data.find('.//DataSet[@TagName="APERTURE"]')
conductivity = data.find('.//DataSet[@TagName="GMFACE_FRACTURE_CONDUCTIVITY"]')

# Parse the text data as a pandas DataFrame
cond_data = pn.read_csv(StringIO(conductivity.text), sep='\t\t\t\t', header=None, names=['id_face', 'cond'])

# Modify the data
cond_data.loc[:, 'cond'] = cond_data.cond * 1.1
# Enforce data types
cond_data = cond_data.astype({'id_face':str, 'cond': str})

# Return the modified data to the text format
b = cond_data.values.tolist()
step_1 = ['\t\t\t\t'.join(elm) for elm in b]
step_2 = '\n\t\t\t\n\t\t\t' + '\n\t\t\t'.join(step_1) + '\n\t\t\t\n\t\t\t'

conductivity.text = step_2
tree.write(files_path + 'test_xml.xml')

