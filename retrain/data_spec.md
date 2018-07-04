# Data Specifications

## CR Code
D01\_P00001234\_P01\_S02
- Database Index: 01
- Patient Index: 00001234
- Phase: 01
- Slice Index: 02

Note that the slice index must be maintained in-order. (oap-obs, obs-oap optional)

## Training Images
Images inside `images` directory, ready to retrain with the `cr_learn.py` module.

<cr\_code>\_CP20\_R180<.aug>.jpg
- Cropped 20% on all four sides (10x10 to 6x6)
- Rotated by 180 degress

## Labels
- `oap`: Out of apical
- `ap`: Apical
- `md`: Middle
- `bs`: Basal
- `obs`: Out of basal

## Image Metadata [`cr_metadata.json`]


## Train Results [`results.json`]
[{
	'tfhub_module': url of tfhub module
	'training_steps': int
	'learning_rate': float
	'validation_percentage': float
	'batch_size': int
	'test_accuracy': float
	'training_images': [paths]
	'predictions': explained below
}...]

### results['predictions']
{
	'image_basename': [answer, truth]
	...
}
