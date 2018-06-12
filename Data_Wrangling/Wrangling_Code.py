
# coding: utf-8

# In[6]:

import csv
import codecs
import re
import xml.etree.cElementTree as ET

import cerberus

import schema

OSM_PATH = "southampton_england.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def wrangle_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Wrangling the data"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  
    node_at=['id','user','uid','version','lat','lon','timestamp','changeset']   

    way_at=['id','user','uid','version','timestamp','changeset']
    
    
    if element.tag == 'node':  #for node tags      
        for i in element.attrib:
            if i in node_at:
                node_attribs[i]=element.attrib[i]
        
        tag_data=element.findall('tag') 
        
        for elem in tag_data:
            out={}
            out['id']=element.attrib['id']
            out['value']=elem.attrib['v']
            
            # Q2 Checking / Cleaning blank key values
            if elem.attrib['k']=='':                
                elem.attrib['k']=='undefined'
                
            # Q1 Checking / Cleaning for colon data
            if elem.attrib['k'].find(':') == -1:  #if there is no colon found              
                out['type']='regular'
                
                #Q4 Filtering & Clustering problematic characters
                if PROBLEMCHARS.search(elem.attrib['k']):
                    out['key']='ProblemChars'
                else:
                    out['key']=elem.attrib['k']
                    
            # Q1 Checking / Cleaning for colon values    
            if not elem.attrib['k'].find(':') == -1: #If there is a colon
                pos=elem.attrib['k'].find(':') # Index of the position of the colon              
                out['type']=elem.attrib['k'][:pos] #Type contain data until colon
                
                #Q4 Filtering & Clustering problematic characters
                if PROBLEMCHARS.search(elem.attrib['k']):
                    out['key']='ProblemChars'
                else:
                    out['key']=elem.attrib['k'][pos+1:] #Key contains data after colon
            
            tags.append(out) #Adding the dictionary for this element into the overall structure
            
    if element.tag == 'way':# for way tags        
        for i in element.attrib:
            if i in way_at:
                way_attribs[i]=element.attrib[i]
                
        tag_data=element.findall('tag')
        
        for elem in tag_data:
            out={}
            out['id']=element.attrib['id']
            out['value']=elem.attrib['v']
            
            # Q2 Checking / Cleaning blank key values
            if elem.attrib['k']=='':
                elem.attrib['k']=='undefined'                
                
            # Q1 Checking for colon in data 
            if elem.attrib['k'].find(':') == -1: #If there is no colon           
                out['type']='regular'
                
                #Q4 Checking for problematic characters
                if PROBLEMCHARS.search(elem.attrib['k']):
                    out['key']='ProblemChars'
                else:
                    out['key']=elem.attrib['k'] 
                    
             # Q1 Checking for colon in data
            if not elem.attrib['k'].find(':') == -1:#If there is a colon              
                pos=elem.attrib['k'].find(':') #index of the colon                
                out['type']=elem.attrib['k'][:pos]#type contains data till colon
                
                #Q4 Checking for problematic characters
                if PROBLEMCHARS.search(elem.attrib['k']):
                    out['key']='ProblemChars'
                else:
                    out['key']=elem.attrib['k'][pos+1:]#Key is data after colon
            
            #Q3 Checking and consolidating Garage data
            if elem.attrib['k'] == 'building' and  elem.attrib['v'] =='garages':#Checking if building key has garages as value
                out['value']='garage'     
                             
            tags.append(out) #Adding the dictionary for this element into the overall structure
            
        wn_data=element.findall('nd') #for way nodes
        for elem in wn_data:
            out={}
            out['id']=element.attrib['id']
            out['node_id']=elem.attrib['ref']
            out['position']=wn_data.index(elem)
            
            way_nodes.append(out)         
            
    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#          From the case study tutorial              #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Output Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'wb') as nodes_file,          codecs.open(NODE_TAGS_PATH, 'wb') as nodes_tags_file,          codecs.open(WAYS_PATH, 'wb') as ways_file,          codecs.open(WAY_NODES_PATH, 'wb') as way_nodes_file,          codecs.open(WAY_TAGS_PATH, 'wb') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = wrangle_element(element)
            
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


process_map(OSM_PATH, validate=True)

