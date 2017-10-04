# import xml.etree.ElementTree as etree
import pandas as pn
from io import StringIO
from lxml.etree import CDATA
from lxml import etree
import os
import subprocess
import pdb

def multiply_fracture_conductivity(origin_xml, new_xml, multiplier):
    """Function that generates a nex XML with some multiple of the conductivity
    -----------
    Parameters:
        origin_xml: String. Full path to the xml to edit
        new_xml: String. Full path to the new xml
        multipler: Float. Conductivity will be multipled by this parameter
        """

    parser = etree.XMLParser(strip_cdata=False)
    tree = etree.parse(origin_xml, parser)
    root = tree.getroot()

    # Read the relevant features
    data = root.find('Mesh').find('Data')
    aperture = data.find('.//DataSet[@TagName="APERTURE"]')
    conductivity = data.find('.//DataSet[@TagName="GMFACE_FRACTURE_CONDUCTIVITY"]')

    # Parse the text data as a pandas DataFrame
    cond_data = pn.read_csv(StringIO(conductivity.text), sep='\t\t\t\t', header=None, names=['id_face', 'cond'])

    # Modify the data
    cond_data.loc[:, 'cond'] = cond_data.cond * multiplier
    # Enforce data types
    cond_data = cond_data.astype({'id_face':str, 'cond': str})

    pdb.set_trace()
    # Return the modified data to the text format
    b = cond_data.values.tolist()
    step_1 = ['\t\t\t\t'.join(elm) for elm in b]
    step_2 = '\n\t\t\t\n\t\t\t' + '\n\t\t\t'.join(step_1) + '\n\t\t\t\n\t\t\t'
    # Add the CDATA tag to the new string
    step_3 = CDATA(step_2)
    conductivity.text = step_3
    tree.write(new_xml)


def create_dialog_file(output_path, mesh_path, wells_path, save_file_path):
    full_text = """OUTPUT
    {0}
    /

    MESH
    {1}
    /

    SHIFT
    no_rescale
    /

    PERM
    data
    PERM
    eigenvalues_check
    /

    PORO
    -- Method for input of porosity.
    data
    PORO 1
    /

    SCHEME
    TPFA
    /

    FRACTURES
    fractures
    data APERTURE 0 0.00269856 0.00686225 1
    FULL
    /

    EXPORTFLOW
    --multiplier for conversation of volume units
    1
    --question to output permeability
    output_permeability
    --multiplier for conversation of transmissibility units
    0.0085
    /

    EXPORTFLOWCUSTOM
    PRESSURE,SGAS,SWAT,SATNUM,ZMF#*:L,RTEMP,THCROCK
    /

    WELLS
    wells
    {2}
    /

    EXPORTWELLS
    --multiplier for conversation of well index units
    1
    /

    EXPORTGM
    no_geomech
    /
    """.format(output_path, mesh_path, wells_path)
    f = open(save_file_path, 'w')
    f.write(full_text)
    f.close()


def create_wells_file(save_file_path):
    full_text = """1
    W	4697.5	RATE	1600	injector	peacman_vertical
    2
    -100	0	-4680
    100	0	-4680
    1
    1	0	1	0.1	0	111
    """
    f = open(save_file_path, 'w')
    f.write(full_text)
    f.close()


def main():
    # Open the XML file
    # files_path = 'D:\\Users\dorta\Dropbox\Stanford\Research\work' \
    #              'space\\test_run\DiscretizationToolkit\\'
    # save_path = 'D:\\Users\dorta\Dropbox\Stanford\Research\workspace\sandbox\\'
    files_path = 'D:/Users/dorta/Dropbox/Stanford/Research/work' \
                 'space/test_run/DiscretizationToolkit/'
    save_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/'
    
    orig_filename = files_path + 'xDante.xml'

    # Iterate though the multipliers
    for m in range(80, 121, 5):
        # Create new folder for the files
        directory = save_path + '{0}/'.format(m)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the full path of the new files
        new_xml_path = directory + '{0}_mesh.xml'.format(m)
        wells_txt_path = directory + '{0}_wells.txt'.format(m)
        dialog_path = directory + 'dialog.txt'

        # Create dialog.txt file
        create_dialog_file(directory, new_xml_path, wells_txt_path, dialog_path)
        # Create wells.txt file
        create_wells_file(wells_txt_path)
        # Create the mesh xml file
        my_mult = m / 100.0
        multiply_fracture_conductivity(orig_filename, new_xml_path, my_mult)

        # Command line
        args_cmd = ['{0}DISCR2GPRS'.format(save_path), '{0}/dialog.txt'.format(directory)]
        subprocess.run(args=args_cmd, stdout=subprocess.PIPE, shell=True)
        # print(p.stdout)





if __name__ == "__main__":
    main()