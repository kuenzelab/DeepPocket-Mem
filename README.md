# DeepPocket-Mem: Ligand Binding Site Detection at Protein-Membrane Interfaces

DeepPocket-Mem is a 3D convolutional neural network framework for ligand binding site detection at protein-membrane interfaces and segmentation from protein structures. 


## Requirements

[Fpocket](https://github.com/Discngine/fpocket)
[Pytorch](https://pytorch.org/)
[libmolgrid](https://github.com/gnina/libmolgrid)
[Biopython](https://biopython.org/) 


## Dataset Preprocessing

Preparing training data:

1) remove hetero atoms (clean_pdb.py)
2) run fpocket through structures (fpocket -f *_protein.pdb)
3) get candidate pocket centers for all structures (get_centers.py)
4) create .gninatypes files for all structure (gninatype() in types_and_gninatyper.py)
5) make train and test types (make_types.py)
6) create molcache file for training (create_molcache2.py)

Example usage of create_molcache2:

	python create_molcache2.py -c 4 --recmolcache train.molcache2 train.types 

PDB files are parsed to remove hetero atoms, then converted to "gninatypes" files and collected into a "molcache2" file  model training with libmolgrid. "gninatypes" and "molcache2" files are binary files that store a representation of the input protein to be used for gridding the molecule. 

".types" files contain training data points prepared, the first column is the class label, the next three columns are pocket center cordinates (x,y,z) and the final columns contain molecule files required for that datapoint.  

Datasets used for training and testing, model, .types, and .molcache2 files can be downloaded from Zenodo at https://doi.org/10.5281/zenodo.19606062.

## Predicting Binding Site

"predict.py" is a script that can be used for predicting binding sites from a .pdb file. It follows the following procedure:

1) Hetero atom removal (clean_pdb)
2) fpocket run
3) Parsing fpocket output for candidate centers (get_centers)
4) Creating gninatypes and types file for CNN input (types_and_gninatyper)
5) Rerank types input according to CNN score (rank_pockets)
6) Segment shape of top ranked pockets (segment_pockets)

Example usage of predict.py:

    python predict.py -p protein.pdb -c class_best_test_auc_54001.pth.tar -s seg_best_test_IOU_93.pth.tar -r 3


## Training classifier

Example usage of train.py:

	python train.py -m model.py --train_types train.types --test_types test.types --batch_size 8 --iterations 20000 --base_lr 0.0001 --solver Adam --outprefix transfer_run1


## Training segmentation

Example usage of train_segmentation.py:

	python train_segmentation.py --train_types seg_train.types --test_types seg_test.types --train_recmolcache train.molcache2 --test_recmolcache test.molcache2 -b 4 -e 150 -o seg1 --base_lr 0.0001
    






