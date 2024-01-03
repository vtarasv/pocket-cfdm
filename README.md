[ci-image]: https://github.com/vtarasv/pocket-cfdm/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/vtarasv/pocket-cfdm/actions/workflows/ci.yml

<h1 align="center">
<p> Augmenting a training dataset of the generative diffusion model for molecular docking with artificial binding pockets</h1>

 ---
# Article
[doi.org/10.1039/D3RA08147H](https://doi.org/10.1039/D3RA08147H)
# Data preparation
The model is trained to dock small molecules in a predefined binding pocket. 
Therefore, the input PDB file is expected to include only pocket residues.
A general recommendation is to consider all the residues within 5-6 Å of any heavy atom of known ligand (15-30 residues) 
or equivalent pocket sizes for the binding sites defined by other methods. Refer to the `examples/extract_pocket.py` as basic pocket extraction script.
# Usage
## From source code
1. Install the recommended dependencies compatible to your hardware and operating system <br />
   - [Git LFS](https://git-lfs.com/) <br />
   - [Python](https://www.python.org/) >= 3.8 <br />
   - [PyTorch](https://pytorch.org/) >= 2.0 <br />
   - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) <br />
2. Clone the repository, navigate to the cloned folder, pull model weights <br />
`git clone https://github.com/vtarasv/pocket-cfdm.git` <br />
`cd pocket-cfdm/` <br />
`git lfs pull` <br />
3. Install required packages <br />
`pip install -r requirements.txt` <br />
4. Run the inference <br />
`python predict.py --pdb my_pocket.pdb --sdf my_ligands.sdf --save_path my_ligands_docked.sdf --samples 16 --batch_size 16 --no_filter` <br />
An increase of `samples` argument will lead to generation of higher alternative poses per docked molecule (better prediction quality for additional computational cost). <br />
Consider decreasing the `batch_size` if you face GPU memory-related errors. <br />
By default the results include only poses with acceptable quality. The `no_filter` flag allows to write all the generated poses despite their quality. <br />
The first script run will take some time to precompute and save in the cache required data distributions.  
## Docker image
[![CI Status][ci-image]][ci-url]
1. Pull the docker image <br />
`docker pull vtarasv/pocket-cfdm`
2. Run the inference code using docker <br />
`docker run -it --rm --gpus all -v '/home/':'/home/' vtarasv/pocket-cfdm -m predict --pdb /home/user/temp/my_pocket.pdb --sdf /home/user/temp/my_ligands.sdf --save_path /home/user/temp/my_ligands_docked.sdf` <br />