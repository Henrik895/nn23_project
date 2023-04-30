This is a generator for generating training images with labels. The generator samples uniformly over the unique labels.

The generator can be used with following command
```
python generate.py
```
This produces one example and stores the image in images/train folder. To create more examples than one, then run
```
python generate.py --examples_nr 10
```
More info about the possible arguments can be found in the following table. To see the default values of the arguments check out the generate.py where the default arguments can be seen in the parse_opt() function.

| Argument  | Describtion |
| ------------- | ------------- |
| --dim_h  | The dimension which is being used during generating, should be same in which the symbols samples are taken  |
| --dim_w | The dimension which is being used during generating, should be same in which the symbols samples are taken |
| --save_dim_h | Height in which the generated images are saved |
| --save_dim_w | Width in which the generated images are saved |
| --save_as_square | Saves the image as square. Uses save_dim_w as dimension for both. Overwrites save_dim_h |
| --save_as_inverse | Switches black and white pixels when saving |
| --vertical_ratio | Probabilty that generated image is vertical / will switch dim_h and dim_w |
| --examples_nr | Number of images to generate |
| --symbols_dir | Directory in which the sample of tactical tasks are |
| --real_symbols_dir | Directory in which the sample of tactical tasks cut from real films are |
| --real_symbols_clean_dir | Directory in which the sample of tactical tasks cut from real films are |
| --unit_symbols_dir | Directory in which the sample of unit symbols are |
| --extras_dir | Directory in which the sample of extras is |
| --save_images_dir | Directory where to store generated images |
| --save_labels_dir | Directory where to store labels |
| --save_rotations_dir | Directory where to store rotations |
| --real_backgrounds_ratio | Ratio of data with real backgrounds |
| --real_backgrounds_dir | Directory in which the real data backgrounds are |
| --real_symbols_ratio | Ratio of real symbols cut from film |
| --real_symbols_in_real_backgrounds | use real symbols in real backgrounds while generating real backgrounds data |

--real_backgrounds_ratio <ratio>` argument will produce `<ratio>` amount of images with real backgrounds. For this to work `data\real_backgrounds` should contain the real background images with their label files or a custom path can be given with `--real_backgrounds_dir <path>` argument.

`--real_symbols_ratio <ratio>` makes the `<ratio>` amount of symbols to be from the real data. For this to work `data/real_symbols` and `data/real_clean_symbols` should have folder that has the real data symbols example of the hierarchy `data/real_symbols/RealTrainSymbols` and `data/real_clean_symbols/RealTrainSymbolsCleaned` or you can give other paths with `--real_symbols_dir <path>` and  `--real_symbols_clean_dir <path>` but that path should contain at least 1 folder which has the symbols. The names of the real symbols and cleaned version of those should have the same names.

Giving `--real_symbols_in_real_backgrounds` argument will enable generation of real symbols on real backgrounds if real background generation is enabled.
