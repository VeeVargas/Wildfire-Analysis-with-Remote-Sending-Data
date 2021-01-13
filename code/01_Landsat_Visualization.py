import ee
from ee_plugin import Map

#country boundary
Countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
Australia = Countries.filter(ee.Filter.eq('country_na', 'Australia'))

landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1") \
    .filterDate('2019-01-01', '2020-01-01') \
    .filterBounds(Australia)
    
#cloud mask
composite = ee.Algorithms.Landsat.simpleComposite(**{
  'collection': landsat,
  'asFloat': True
})

rgbVis = {'bands': ["B4", "B3", "B2"], 'min':0, 'max': 0.3}
nirVis = {'bands': ["B5", "B4", "B3"], 'min':0, 'max': [0.5, 0.3, 0.3]}

Map.addLayer(composite, rgbVis, "RGB")
Map.addLayer(composite, nirVis, "False Color")
Map.centerObject(Australia)


