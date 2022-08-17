#$ -S /bin/bash
#$ -q short
#$ -l ngpus=1
#$ -l ncpus=6
#$ -l h_vmem=80G
#$ -l h_rt=12:00:00
#$ -M st7ma784@gmail.com
#$ -m beas
source /etc/profile
module add anaconda3/wmlce
source activate $global_storage/conda4
modelcache=$global_storage/data/pretraining
module add git
cd $global_scratch
git clone https://github.com/carlosGarciaHe/MS-COCO-ES.git
cd $global_storage/6DIMCOCO
git pull
export WANDB_SILENT=true
export WANDB_RESUME=auto
export WANDB_CONSOLE='off'
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
export ISHEC=1
# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export wandb='9cf7e97e2460c18a89429deed624ec1cbfb537bc'
python trainagent.py --data_dir $global_scratch --sweep_id rv9zla43

