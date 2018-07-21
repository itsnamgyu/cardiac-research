# Data Specifications

## CR Code
`D01_P00001234_P01_S02`

- Database Index: 01
- Patient Index: 00001234
- Phase: 01
- Slice Index: 02

Note that the slice index must be maintained in-order. (oap-obs, obs-oap optional)


## Labels
- `oap`: Out of apical
- `ap`: Apical
- `md`: Middle
- `bs`: Basal
- `obs`: Out of basal

## Image Metadata
`cr_metadata.json`
```
{
	'D00_P00000101_P00_S00':
	{
		'original_filepath': 'cap_challenge/DET0000101/DET0000101_SA12_ph0.dcm', 
		'original_name': 'DET0000101_SA12_ph0', 
		'label': 'obs'
	}
	...
}
```

## Image Database
Original images inside `cr_database` directory.

Filename format: `<cr_code>.jpg`

## Training Images
Images inside `images` directory, ready for use with the `cr_learn.py` module.

`<cr_code>_CP20_R180<.aug>.jpg`

- Cropped 20% on all four sides (10x10 to 6x6)
- Rotated by 180 degress

## Train Results [`results.json`]
```
[
	{
	'tfhub_module': url of tfhub module
	'training_steps': int
	'learning_rate': float
	'validation_percentage': float
	'batch_size': int
	'test_accuracy': float
	'training_images': [paths]
	'predictions': explained below
	}
	...
]
```

### results['predictions']
```
{
	'image_basename'
	{
		'prediction': 'oap', 
		'truth': 'oap', 
		'percentages': {'oap': float (0-1)...}
	}
}
```