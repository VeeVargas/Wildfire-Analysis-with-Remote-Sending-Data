USER_NAME = 'vee_vargas'
OUTPUT_BUCKET = 'aus_fire_bucket'
FOLDER = 'training_data'
TRAINING_BASE = 'training_patches'
TESTING_BASE = 'testing_patches'
VAL_BASE = 'val_patches'

KERNEL_SIZE = 256

# create an EE kernel opject from the kernel size
list = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list, KERNEL_SIZE)
kernel = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)

# Landsat 
L8SR = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
# Use these bands 
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
LABEL = 'landcover'
FEATURE_NAMES = list(BANDS)
FEATURE_NAMES.append(LABEL)

# Export imagery in this region.
EXPORT_REGION = ee.Geometry.Rectangle([-122.7, 37.3, -121.8, 38.00])
                    
# Cloud masking function.
def maskL8sr(image):
  cloudShadowBitMask = ee.Number(2).pow(3).int()
  cloudsBitMask = ee.Number(2).pow(5).int()
  qa = image.select('pixel_qa')
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
  return image.updateMask(mask).select(BANDS).divide(10000)
    
# upsample the features (bilinear upsample)
def decoder_block(input_tensor, concat_tensor=None, nFilters=512,nConvs=2,i=0,name_prefix="decoder_block"):
    deconv = input_tensor
    for j in range(nConvs):
        deconv = layers.Conv2D(nFilters, 3, activation='relu',
                               padding='same',name=f"{name_prefix}{i}_deconv{j+1}")(deconv)
        deconv = layers.BatchNormalization(name=f"{name_prefix}{i}_batchnorm{j+1}")(deconv)
        if j == 0:
            if concat_tensor is not None:
                 deconv = layers.concatenate([deconv,concat_tensor],name=f"{name_prefix}{i}_concat")
            deconv = layers.Dropout(0.2, seed=0+i,name=f"{name_prefix}{i}_dropout")(deconv)
    
    up = layers.UpSampling2D(interpolation='bilinear',name=f"{name_prefix}{i}_upsamp")(deconv)
    return up

def normalized_difference(a, b):
  """Compute normalized difference of two inputs.

  Compute (a - b) / (a + b).  If the denomenator is zero, add a small delta.

  Args:
    a: an input tensor with shape=[1]
    b: an input tensor with shape=[1]

  Returns:
    The normalized difference as a tensor.
  """
  nd = (a - b) / (a + b)
  nd_inf = (a - b) / (a + b + 0.000001)
  return tf.where(tf.math.is_finite(nd), nd, nd_inf)

def add_NDVI(features, label):
  """Add NDVI to the dataset.
  Args:
    features: a dictionary of input tensors keyed by feature name.
    label: the target label

  Returns:
    A tuple of the input dictionary with an NDVI tensor added and the label.
  """
  features['NDVI'] = normalized_difference(features['B5'], features['B4'])
  return features, label


























