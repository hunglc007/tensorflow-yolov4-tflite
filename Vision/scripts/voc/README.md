# VOC Dataset

### Download

```bash
$ bash get_voc2012.sh
```

### Make names for VOC.

```bash
$ python voc_make_names.py [--anno_dir {Annotation directory}] [--output {OUTPUT_NAME}]

# example
$ python voc_make_name.py

$ python voc_make_name.py --anno_dir ../../data/voc/anno --output ../../data/classes/voc.names
```

### Convert VOC Dataset.

```bash
$ python voc_convert.py [--image_dir {Image directory}] [--anno_dir {Annotation directory}] [--train_list_txt {Path of Train list file}] [--val_list_txt {Path of Validation list file}] [--classes {Path of Classes file}] [--train_output {Path of Output file For Train}] [--val_output {Path of Output file For Val}]

#example
$ python voc_convert.py
```
